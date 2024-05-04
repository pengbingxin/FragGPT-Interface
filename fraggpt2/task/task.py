import os
import torch
import math
import logging
import transformers
# import torch._dynamo as dynamo
from transformers.pytorch_utils import Conv1D
from models.ema import ModelEMA
from torch.utils.data import DataLoader

if transformers.__version__.startswith('4.33'):
    from peft import LoraConfig, TaskType, get_peft_model
if transformers.__version__.startswith('4.26'):
    from transformers import GPT2Config, PfeifferConfig

from transformers import (
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from utils.utils import save_config, get_parameter_number

def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = Conv1D
    # cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            print(name)
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

class Task(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.model = None
        self.ema = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.test_datasets = None
        self.loss = None
        self.optimizer = None
        self.lr_scheduler = None
        self.max_train_steps = None
        self.accelerator = None
        self.logger = None
        self.wandb = None
        self.tokenizer = None

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg, **kwargs)

    def set(self, accelerator, logger, wandb):
        self.accelerator = accelerator
        self.logger = logger
        self.wandb = wandb

    def build_dataset(self, cfg):
        import datasets
        self.train_datasets = datasets.build_datasets(cfg, mode='train')
        self.valid_datasets = datasets.build_datasets(cfg, mode='valid')

        self.tokenizer = self.train_datasets.tokenizer
        self.train_dataloader = DataLoader(
            self.train_datasets,
            collate_fn=self.train_datasets.collator,
            batch_size=cfg.SOLVER.TRAIN_BSZ,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.SOLVER.NUM_WORKERS,
            drop_last=True,
        )

        self.valid_dataloader = DataLoader(
            self.valid_datasets,
            collate_fn=self.valid_datasets.collator,
            batch_size=cfg.SOLVER.VALID_BSZ,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.SOLVER.NUM_WORKERS,
            drop_last=False,
        )

    def build_model(self, cfg):
        import models
        tokenizer = self.tokenizer
        self.model = models.build_model(cfg, self)
        # self.model = torch.compile(self.model)
        # self.model = dynamo.optimize("inductor")(self.model)

        if self.cfg.MODEL.USE_MODEL_CKPT:
            pretrain_path = os.path.join(self.cfg.MODEL.CHECKPOINT_PATH, self.cfg.MODEL.MODEL_NAME)
            assert os.path.exists(pretrain_path), 'checkpoint no exists! '
            pretrained_dict = torch.load(pretrain_path, map_location='cpu')
            model_dict = self.model.state_dict()
            if 'model' in pretrained_dict:
                pretrained_dict = {k: v for k, v in pretrained_dict['model'].items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}

            total = len(model_dict.keys())
            rate = sum([1 for k in model_dict.keys() if k in pretrained_dict]) / total
            print('参数加载率：', rate)
            print([k for k in pretrained_dict.keys() if k not in model_dict])
            print([k for k in model_dict.keys() if k not in pretrained_dict])

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            self.logger.info(f"  =====================================================================================")
            self.logger.info(f"  Load pretrained model from {pretrain_path}")
            self.logger.info(f"  =====================================================================================")


        # if self.cfg.MODEL.ADAPTER.use_adapter:
        #     adapter_config = PfeifferConfig(
        #         original_ln_before=True, original_ln_after=True, residual_before_ln=True,
        #         adapter_residual_before_ln=False, ln_before=False, ln_after=False,
        #         mh_adapter=False, output_adapter=True, non_linearity=self.cfg.MODEL.ADAPTER.ACTIVATION,
        #         reduction_factor=self.cfg.MODEL.ADAPTER.REDUCTION_FACTOR, cross_adapter=False)
        #     adapter_name = self.cfg.MODEL.ADAPTER.adapter_name
        #     self.model.add_adapter(adapter_name, config=adapter_config)
        #     self.model.train_adapter(adapter_name)
        #     self.model.set_active_adapters(adapter_name)
        #     if cfg.MODEL.ADAPTER.OPEN_HEAD:
        #         for p in self.model.lm_head.parameters():
        #             p.requires_grad = True

        if self.cfg.MODEL.PEFT.LoRA:
            print('use LoRA')
            # target_modules = find_all_linear_names(self.model)
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                # target_modules=target_modules,
                inference_mode=False,
                r=self.cfg.MODEL.PEFT.r,
                lora_alpha=self.cfg.MODEL.PEFT.lora_alpha,
                bias="none",
                lora_dropout=self.cfg.MODEL.PEFT.lora_dropout)

            # self.model = inject_adapter_in_model(lora_config, self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.logger.info(f"  -*- -*- -*- -*- -*- -*- 参数量 -*- -*- -*- -*- -*- -*- -*- -*- ")
        self.logger.info(get_parameter_number(self.model))
        self.logger.info(f"   -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- ")

        if cfg.MODEL.USE_EMA:
            self.ema = ModelEMA(self.model)

    def build_optim(self, cfg):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params, lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.99), eps=1e-6, no_deprecation_warning=True)

        if self.ema is not None:
            self.ema.ema = self.accelerator.prepare(self.ema.ema)

        self.model, self.optimizer, self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(
            model,
            optimizer,
            self.train_dataloader,
            self.valid_dataloader,
        )

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / cfg.SOLVER.GRADIENT_ACC)
        print('num_update_steps_per_epoch: ', num_update_steps_per_epoch)
        max_train_steps = cfg.SOLVER.MAX_EPOCHS * num_update_steps_per_epoch
        num_warmup_steps = cfg.SOLVER.WARMUP_STEP_RATIO * max_train_steps

        self.lr_scheduler = get_scheduler(
            name=cfg.SOLVER.LR_SCHEDULER,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps
        )

        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7, last_epoch=-1)
        self.max_train_steps = max_train_steps
