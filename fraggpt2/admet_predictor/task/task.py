import os
import torch
import math
import logging

import numpy as np

from torch.utils.data import DataLoader, Subset
from transformers import (
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from Transformer.models.ema import ModelEMA
from sklearn.model_selection import StratifiedKFold

class Task(object):
    def __init__(self, args, **kwargs):
        self.args = args
        self.model = None
        self.ema = None
        self.train_val_datasets = None
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
        self.train_index_list = None
        self.valid_index_list = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, **kwargs)

    def set(self, accelerator, logger):
        self.accelerator = accelerator
        self.logger = logger

    def get_trainer_config(self):
        assert self.model != None
        assert self.train_dataloader != None
        assert self.valid_dataloader != None
        assert self.loss != None
        assert self.optimizer != None
        assert self.lr_scheduler != None
        assert self.max_train_steps != None
        if self.args.config not in ['BEMT_Pretrain']:assert self.test_dataloader != None

        return {
            'loss_module': self.loss,
            'accelerator': self.accelerator,
            'model': self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'ema': self.ema
        }

    def build_dataset(self, args, i, K):
        import datasets
        def split_train_valid(datasets, K):
            assert args.task_type in ['cls', 'reg']
            num_data = len(datasets)
            train_index_list, valid_index_list = None, None
            if args.task_type == 'cls':
                temp = np.array([[i, int(datasets[i]['net_input.batched_data'].y)] for i in range(num_data)])
                X, y = temp[:, 0], temp[:, 1]
                skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=args.seed).split(X, y)
                train_index_list = []
                valid_index_list = []
                for train_index, valid_index in skf:
                    train_index_list.append(train_index)
                    valid_index_list.append(valid_index)

            elif args.task_type == 'reg':
                idx = list(range(num_data))
                # np.random.seed(args.seed)
                np.random.shuffle(idx)
                valid_index_list = [idx[int((i/K)*num_data): int(((i+1)/K)*num_data)] for i in range(K)]
                train_index_list = [[k for k in idx if k not in valid_index_list[j]] for j in range(K)]
            return train_index_list, valid_index_list

        if self.train_val_datasets is None:
            self.train_val_datasets = datasets.build_datasets(args, mode='train')
        if self.train_index_list is None or self.valid_index_list is None:
            self.train_index_list, self.valid_index_list = split_train_valid(self.train_val_datasets, K)
        train_idx = self.train_index_list[i]
        valid_idx = self.valid_index_list[i]

        self.train_datasets = Subset(self.train_val_datasets, indices=train_idx)
        self.valid_datasets = Subset(self.train_val_datasets, indices=valid_idx)
        if self.args.test:
            self.test_datasets = datasets.build_datasets(args, mode='test')
        self.train_dataloader = DataLoader(
            self.train_datasets,
            collate_fn=self.train_val_datasets.collater,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=False,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.valid_dataloader = DataLoader(
            self.valid_datasets,
            collate_fn=self.train_val_datasets.collater,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        if self.args.test:
            self.test_dataloader = DataLoader(
                self.test_datasets,
                collate_fn=self.test_datasets.collater,
                batch_size=args.batch_size,
                shuffle=False,
                pin_memory=False,
                num_workers=args.num_workers,
                drop_last=False,
            )

    def build_model(self, args):
        from Transformer import models
        self.model = models.build_model(args, self)
        if self.args.usr_pretrain:
            pretrained_dict = torch.load(self.args.checkpoint, map_location='cpu')
            model_dict = self.model.state_dict()
            if 'model' in pretrained_dict:
                pretrained_dict = {k: v for k, v in pretrained_dict['model'].items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}

            ###########
            # if 'encoder.lm_head_transform_weight.weight' in pretrained_dict:
            #     pretrained_dict.pop('encoder.lm_head_transform_weight.weight')
            ###########
            total = len(model_dict.keys())
            rate = sum([1 for k in model_dict.keys() if k in pretrained_dict]) / total
            print('参数加载率：', rate)
            # print([k for k in pretrained_dict.keys() if k not in model_dict])
            print([k for k in model_dict.keys() if k not in pretrained_dict])

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            self.logger.info(f"  =====================================================================================")
            self.logger.info(f"  Load pretrained model from {self.args.checkpoint}")
            self.logger.info(f"  =====================================================================================")

        if args.freeze:
            for (name, param) in self.model.named_parameters():
                param.requires_grad = False

            no_freeze = ['encoder.task_head.weight', 'encoder.task_head.bias', 'encoder.smi_task_head.weight',
                         'encoder.smi_task_head.bias', 'encoder.graph_task_head.weight', 'encoder.graph_task_head.bias']
            for (name, param) in self.model.named_parameters():
                if 'fusion' in name or name in no_freeze:
                    param.requires_grad = True

            no_freeze_list = []
            for (name, param) in self.model.named_parameters():
                if param.requires_grad == True:
                    no_freeze_list.append(name)
            print(' 没有被冻结的参数如下：', no_freeze_list)

        if args.use_ema:
            self.ema = ModelEMA(self.model)

    def build_loss(self, args):
        from Transformer import criterions
        self.loss = criterions.build_losses(args, self)

    def build_optim(self, args):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params, lr=args.learning_rate, betas=(0.9, 0.99), eps=1e-6)

        if self.ema is not None:
            self.ema.ema = self.accelerator.prepare(self.ema.ema)

        self.model, self.optimizer, self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.accelerator.prepare(
            model,
            optimizer,
            self.train_dataloader,
            self.valid_dataloader,
            self.test_dataloader,
        )

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        print('num_update_steps_per_epoch: ', num_update_steps_per_epoch)
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        num_warmup_steps = args.num_warmup_steps * max_train_steps

        self.lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps
        )
        self.max_train_steps = max_train_steps
