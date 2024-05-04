import torch
import os
import json
import yaml
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from datasets.GraphDatasets import load_dataset
from Transformer.models.transformer_m import TransformerMModel
from Transformer.models.cress_admet import MultiTaskCress
from datasets import get_task_names

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

class InferenceADMET:
    def __init__(self, args, model_n, gpu=False):
        # set
        self.args = args
        self.gpu = gpu
        self.device = torch.device('cpu')
        if self.gpu:
           self.device = torch.device('cuda:0')

        if model_n.split('_')[0] == 'Cress':
            self.model = MultiTaskCress(args).to(self.device)
        elif model_n.split('_')[0] in ['GraphTransformer', 'GraphTransformer18']:
            self.model = TransformerMModel.build_model(args, None).to(self.device)
        else:
            raise ValueError(' No exist mode in Cress or GraphTransformer. ')

        self.model = self.load_ckpt(self.model, os.path.join(os.getcwd(), 'save', model_n + '.pt'))

        self.dataset = load_dataset(args, mode='test')
        self.dataloader = DataLoader(
            self.dataset,
            collate_fn=self.dataset.collater,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=False,
        )

    def load_ckpt(self, model, ckpt_path):
        print(f'load ckpt from {ckpt_path}')
        pretrained_dict = torch.load(ckpt_path)
        model_dict = model.state_dict()

        total = len(model_dict.keys())
        rate = sum([1 for k in model_dict.keys() if k in pretrained_dict]) / total
        print('参数加载率：', rate)
        # pretrained_dict = {('.'.join(k.split('.')[1:])): v for k, v in pretrained_dict.items()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=True)
        model.eval()
        return model

    def run(self):
        outputs = []
        targets = []
        masks = []
        smiles = []
        pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        for idx, batch in pbar:
            if self.gpu:
                batch_net_input = {}
                for k, v in batch['net_input']['batched_data'].items():
                    if v is None:continue
                    if type(v) != list:
                        batch_net_input[k] = v.to(self.device)
                    else:
                        batch_net_input[k] = v
                batch['net_input']['batched_data'] = batch_net_input
                batch['target'] = batch['target'].to(self.device)
            output = self.model(**batch['net_input']).logits
            outputs.append(output.cpu().detach().numpy())
            targets.append(batch['target'].cpu().detach().numpy())
            masks.append(batch['net_input']['batched_data']['task_mask'].cpu().detach().numpy())
            smiles += batch["net_input"]["batched_data"]["smiles"]

        y_pred, y_true, masks, smiles = np.concatenate(outputs), np.concatenate(targets), np.concatenate(masks), np.array(smiles)
        task_info = get_task_names()
        id2task = {i: n for i, n in enumerate(task_info[1])}
        num_task = task_info[2] + task_info[3]
        df_pred = pd.DataFrame(y_pred, columns=['pred_' + id2task[i] for i in range(num_task)])
        df_true = pd.DataFrame(y_true, columns=['target_' + id2task[i] for i in range(num_task)])
        df_mask = pd.DataFrame(masks, columns=['mask_' + id2task[i] for i in range(num_task)])
        df_smi = pd.DataFrame(smiles, columns=['smi'])
        df = pd.concat([df_smi, df_pred, df_true, df_mask], axis=1)

        df = df.groupby('smi').mean()
        df['smi'] = df.index
        df = df.reset_index(drop=True)
        return np.array(df.iloc[:, :num_task]), np.array(df.iloc[:, num_task: 2*num_task]), np.array(df.iloc[:, 2*num_task: 3*num_task]), df.iloc[:, 3*num_task:]

def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble inference for GraphTransformer and Cress")
    parser.add_argument(
        "--config",
        type=str,
        default='GraphTransformer_1,GraphTransformer_2,GraphTransformer_3,GraphTransformer18_4,Cress_1,Cress_2',     # Cress2: task_weight   GraphTransformer_2: 蒸馏Cress   GraphTransformer_3: task_weight_bs64    GraphTransformer_4: L18
        help="Selected a config for this task."
    )
    parser.add_argument(
        "--weight",
        type=list,
        default=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
        help="Selected a config for this task."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='0'
    )
    parser.add_argument("--nni", type=bool, default=False, help="nni parameter search")
    args = parser.parse_args()
    return args, parser

def parse_cfg(parser, cfg, para=None):
    if type(cfg) == dict:
        for k, v in cfg.items():
            parse_cfg(parser, cfg[k], k)
    else:
        parser.add_argument(f'--{para}', default=cfg)
    args = parser.parse_args()
    return args

def reset_args(mode, parser):
    file = open(os.path.join('configs', f'{mode}.yml'), 'r', encoding="utf-8")
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    args = parse_cfg(parser, cfg)
    return args

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_mean_std(args, task_name):
    with open(os.path.join(args.data_path, 'admet_mean_std.json'), 'r') as f:
        mean_std = json.load(f)
    mean = mean_std[task_name]['mean']
    std = mean_std[task_name]['std']
    return mean, std

def cal_metrics(args, logits, targets, masks):
    num_task = logits.shape[1]
    task_info = get_task_names()
    _, task_name, num_cls, num_reg, _ = task_info
    result = {}
    for i in range(num_task):
        if i < num_cls:
            result[task_name[i]] = roc_auc_score(targets[:, i][masks[:, i].astype(np.bool_)], sigmoid(logits[:, i][masks[:, i].astype(np.bool_)]))
        else:
            mean, std = get_mean_std(args, task_name=task_name[i])
            assert mean != None
            assert std != None
            new_y_pred_norm = logits[:, i][masks[:, i].astype(np.bool_)] * std + mean
            result[task_name[i]] = np.sqrt(np.mean((targets[:, i][masks[:, i].astype(np.bool_)] - new_y_pred_norm) ** 2))
    return result

def main():
    args, parser = parse_args()
    mode = args.config.split(',')
    outputs = None
    targets = None
    masks = None
    smiles = []
    for m, w in zip(mode, args.weight):
        parser = argparse.ArgumentParser(description=f"{m}")
        args = reset_args(m.split('_')[0], parser)
        args.config = m
        Executor = InferenceADMET(args, m, gpu=True)
        pred, targ, mask, smile = Executor.run()
        smiles.append(smile)
        if outputs is None:
            outputs = w * pred
            targets = targ
            masks = mask
        else:
            outputs += w * pred
    smi_df = pd.concat(smiles, axis=1)
    print(smi_df)
    result = cal_metrics(args, outputs, targets, masks)
    print(result)


if __name__ == '__main__':
    main()