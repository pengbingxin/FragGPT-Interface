import torch
import os
import nni
import random
import numpy as np
import copy

from task import Task
from tqdm import tqdm

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class Trainer():
    def __init__(self, task):
        self.task = task
        self.args = task.args

        self.accelerator = task.accelerator
        self.logger = task.logger

        self.train_datasets = task.train_datasets
        self.valid_datasets = task.valid_datasets
        self.test_datasets = task.test_datasets
        self.train_dataloader = task.train_dataloader
        self.valid_dataloader = task.valid_dataloader
        self.test_dataloader = task.test_dataloader
        self.max_train_steps = task.max_train_steps

        self.model = task.model
        self.model0 = copy.deepcopy(task.model)
        for p in self.model0.parameters():
            p.requires_grad_(False)
        self.ema = task.ema
        self.loss = task.loss
        self.accelerator = task.accelerator
        self.optimizer = task.optimizer
        self.lr_scheduler = task.lr_scheduler

        self.para_dic = {
            'loss_module': self.loss,
            'accelerator': self.accelerator,
            'model': self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
        }


    def train(self, i):
        # Train!
        total_batch_size = self.args.batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_datasets)} + {len(self.valid_datasets)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.batch_size}")
        self.logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.max_train_steps), disable=not self.accelerator.is_local_main_process)

        para_dic = self.para_dic
        para_dic['gradient_accumulation_steps'] = self.args.gradient_accumulation_steps
        para_dic['progress_bar'] = progress_bar
        para_dic['args'] = self.args

        best_valid_metrics = 0 if self.args.task_type == 'cls' else float('inf')
        best_valid_reg_metrics = float('inf')
        best_test_cls_reg_metrics = {}

        val_metrics = None
        for epoch in range(self.args.num_train_epochs):
            seed_everything(self.args.seed + epoch)
            # train one epoch
            self.train_epoch(self.train_dataloader, val_metrics=val_metrics, **para_dic)
            # eval
            model = self.ema.ema if self.ema is not None else self.model
            val_metrics, eval_loss = self.valid(epoch, para_dic['loss_module'], self.accelerator, model, self.valid_dataloader, self.args)
            # test
            test_metrics = None
            if self.args.test:
                test_metrics = self.test(epoch, para_dic['loss_module'], self.accelerator, model, self.test_dataloader)
            # metrics
            if self.args.task_type == 'cls':
                if val_metrics > best_valid_metrics:
                    best_valid_metrics = val_metrics
                    if test_metrics is not None:
                        best_test_cls_reg_metrics['best_metrics'] = test_metrics
                        best_test_cls_reg_metrics['epoch'] = epoch
                    self.accelerator.wait_for_everyone()
                    self.save_model(self.model, Kfold=i, save_dict=True, methed='unlast')
            else:
                if val_metrics < best_valid_metrics:
                    best_valid_metrics = val_metrics
                    if test_metrics is not None:
                        best_test_cls_reg_metrics['best_metrics'] = test_metrics
                        best_test_cls_reg_metrics['epoch'] = epoch
                    self.accelerator.wait_for_everyone()
                    self.save_model(self.model, Kfold=i, save_dict=True, methed='unlast')

        last_epoch_metrics = {'last_metrics': test_metrics}
        # save
        self.accelerator.wait_for_everyone()
        self.save_model(self.model, Kfold=i, save_dict=True, methed='last')

        if self.args.test:
            self.logger.info(f"  =====================================================================================")
            self.logger.info(f"  The best result of test for valid is {best_test_cls_reg_metrics}")
            self.logger.info(f"  The   last   epoch   metrics   is    {last_epoch_metrics}")
            self.logger.info(f"  =====================================================================================")

        model_dict = self.model0.state_dict()
        self.model.load_state_dict(model_dict, strict=True)


    def save_model(self, model, Kfold, save_dict=True, methed='last'):
        unwrapped_model = self.accelerator.unwrap_model(model)
        if methed != 'last':
            save_file_name = f'best_model_for_valid_{Kfold}.pt'
        else:
            save_file_name = f'last_epoch_model_{Kfold}.pt'
        torch.save(unwrapped_model.state_dict() if save_dict else unwrapped_model, os.path.join('save', self.args.output_dir, save_file_name))

    def train_epoch(
            self,
            train_dataloader,
            val_metrics,
            loss_module,
            gradient_accumulation_steps,
            accelerator,
            model,
            optimizer,
            lr_scheduler,
            progress_bar,
            args,
    ):
        model.train()

        loss_values = []
        for step, item in enumerate(train_dataloader):
            self.train_step(step, gradient_accumulation_steps, accelerator, model, train_dataloader, item, optimizer,
                       lr_scheduler, progress_bar, loss_values, loss_module, args, val_metrics)

    def train_step(
            self,
            step,
            gradient_accumulation_steps,
            accelerator,
            model,
            train_dataloader,
            item,
            optimizer,
            lr_scheduler,
            progress_bar,
            loss_values,
            loss_module,
            args,
            val_metrics=None
    ):

        loss, sample_size, logging_output = loss_module(model, item, split='train', val_metrics=val_metrics)

        loss = loss / gradient_accumulation_steps

        loss_values.append(loss.item())

        accelerator.backward(loss)

        # clip gradient
        if args.clip_grad is not False:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if self.ema is not None:
                self.ema.update(model)

            total_loss_value = sum(loss_values)

            progress_bar.set_description(
                f'loss {total_loss_value:.4f}, lr:{optimizer.param_groups[0]["lr"]:.7f} ',
                refresh=False)
            progress_bar.update(1)

            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss_values.clear()

    def valid(self, epoch, loss_module, accelerator, model, eval_dataloader, args):

        model.eval()
        with torch.no_grad():
            losses = []
            logging_outputs = []
            regressive_loss = []

            for step, item in enumerate(eval_dataloader):
                loss, sample_size, logging_output = loss_module(model, item, split='valid')

                logging_outputs.append(logging_output)
                losses.append(loss.unsqueeze(0))
                regressive_loss.append(logging_output['valid_regressive_loss'].unsqueeze(0))

            eval_loss = torch.cat(losses).mean().item()
            eval_regressive_loss = 0
            if len(regressive_loss) > 0:
                eval_regressive_loss = torch.cat(regressive_loss).mean().item()

        # self.logger.info(f"epoch: {epoch}, eval_loss: {eval_loss}, eval_regressive_loss: {eval_regressive_loss}")

        metrics = loss_module.reduce_metrics(logging_outputs, self.logger, split='valid', args=self.args, loss_module=loss_module)

        if type(metrics) == dict:
            from datasets import get_task_names
            admet_all_task_names, selected_all_task_name, num_cls, num_reg, selected_ids = get_task_names()
            mean_cls_metrics = np.mean([metrics[c] for c in selected_all_task_name[:num_cls] if c in metrics])
            mean_reg_metrics = np.mean([metrics[c] for c in selected_all_task_name[num_cls:] if c in metrics])
            return {'mean_cls_metrics': mean_cls_metrics, 'mean_reg_metrics': mean_reg_metrics, 'all_task_metrics': metrics}, eval_loss
        else:
            return metrics, eval_loss


    def test(self, epoch, loss_module, accelerator, model, eval_dataloader):

        model.eval()
        with torch.no_grad():
            losses = []
            logging_outputs = []
            regressive_loss = []
            for step, item in enumerate(eval_dataloader):
                loss, sample_size, logging_output = loss_module(model, item, split='test')
                logging_outputs.append(logging_output)
                losses.append(loss.unsqueeze(0))

            test_loss = torch.cat(losses).mean().item()
            test_regressive_loss = 0
            if len(regressive_loss) > 0:
                test_regressive_loss = torch.cat(regressive_loss).mean().item()

        metrics = loss_module.reduce_metrics(logging_outputs, self.logger, split='test', args=self.args, loss_module=loss_module)
        if type(metrics) == dict:
            from datasets import get_task_names
            admet_all_task_names, selected_all_task_name, num_cls, num_reg, selected_ids = get_task_names()
            mean_cls_metrics = np.mean([metrics[c] for c in selected_all_task_name[:num_cls] if c in metrics])
            mean_reg_metrics = np.mean([metrics[c] for c in selected_all_task_name[num_cls:] if c in metrics])
            return {'mean_cls_metrics': mean_cls_metrics, 'mean_reg_metrics': mean_reg_metrics, 'all_task_metrics': metrics}
        else:
            return metrics
