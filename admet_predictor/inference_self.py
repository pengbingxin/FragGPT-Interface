import torch
import os
import json
import yaml
import argparse
import sys
sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), 'admet_predictor'))

import pandas as pd
import numpy as np

from tqdm import tqdm
from admet_predictor.datasets import get_task_names
from admet_predictor.datasets.GraphDatasets import load_dataset
from admet_predictor.Transformer.models.bemt2 import BEMT
# from admet_predictor.Transformer.models.transformer_m import TransformerMModel
# from admet_predictor.Transformer.models.cress_admet import MultiTaskCress
from torch.utils.data import DataLoader
# from inference import InferenceADMET


def parse_args():
    parser = argparse.ArgumentParser(description="ADMET project")
    parser.add_argument(
        "--config",
        type=str,
        default='task',
        help="Selected a config for this task."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default='0'
    )
    parser.add_argument("--nni", type=bool, default=False, help="nni parameter search")
    parser.add_argument("--kk", type=int, default=-1)
    args = parser.parse_args()
    return args, parser


def parse_cfg(parser, cfg, para=None):
    if type(cfg) == dict:
        for k, v in cfg.items():
            parse_cfg(parser, cfg[k], k)
    else:
        parser.add_argument('--{}'.format(para), default=cfg)
    args = parser.parse_args()
    return args

class TransformerInference():
    def __init__(self, args, gpu=False):
        # set
        self.args = args
        self.gpu = gpu
        self.device = torch.device('cpu')
        if self.gpu:
            self.device = torch.device('cuda:0')

        model_name_base = args.load_ckpt_method
        n_models_list = os.listdir(os.path.join(os.getcwd(), 'admet_predictor/save', args.output_dir))
        self.n_models = []
        if args.is_kfold:
            for i in range(args.K):
                model_file_name = model_name_base + '_{}.pt'.format(i)
                assert model_file_name in n_models_list, \
                    '模型不存在，请确保模型参数在指定路径，或者请检查参数K是否与训练时匹配'
                ckpt_path = os.path.join(os.path.join(os.getcwd(), 'save', args.output_dir), model_file_name)
                self.n_models.append(self.load_ckpt(BEMT.build_model(args, None).to(self.device), ckpt_path))
        else:
            model_dir = os.path.join(os.getcwd(), 'admet_predictor/save', args.output_dir)
            for name in os.listdir(model_dir):    
                # model_file_name = 'BEMT_Finetune_49.pt'
                if name in ['frag_distil_cls1_reg3_period_admet_smiles_aug.pt']:
                    model_file_name = name
                    ckpt_path = os.path.join(os.path.join(os.getcwd(), 'admet_predictor/save', args.output_dir), model_file_name)
                    self.n_models.append(self.load_ckpt(BEMT.build_model(args, None).to(self.device), ckpt_path))
                elif name in ['GraphTransformer_2.pt_', 'GraphTransformer18_1.pt_']:
                    if name.split('_')[0] == 'GraphTransformer18':
                        args.encoder_layers = 18
                    else:
                        args.encoder_layers = 12
                    model_file_name = name
                    ckpt_path = os.path.join(os.path.join(os.getcwd(), 'admet_predictor/save', args.output_dir), model_file_name)
                    self.n_models.append(self.load_ckpt(TransformerMModel.build_model(args, None).to(self.device), ckpt_path))
                elif name in ['Cress_1.pt_']:
                    model_file_name = name
                    ckpt_path = os.path.join(os.path.join(os.getcwd(), 'admet_predictor/save', args.output_dir), model_file_name)
                    self.n_models.append(self.load_ckpt(MultiTaskCress.build_model(args, None).to(self.device), ckpt_path))
                    

        self.dataset = load_dataset(args, split='test')
        self.dataloader = DataLoader(
            self.dataset,
            collate_fn=self.dataset.collater,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
            drop_last=False,
        )
        self.task_info = get_task_names()
        self.id2task = {i: n for i, n in enumerate(self.task_info[1])}
        self.num_cls, self.num_reg = self.task_info[2], self.task_info[3]
        self.mean_std = self.get_mean_std()


    def load_ckpt(self, model, ckpt_path):
        print(f'load ckpt from {ckpt_path}')
        pretrained_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        total = len(model_dict.keys())
        rate = sum([1 for k in model_dict.keys() if k in pretrained_dict]) / total
        print('参数加载率：', rate)
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        model.eval()
        return model

    def get_mean_std(self):
        with open('admet_predictor/admet_mean_std.json', 'r') as f:
            mean_std = json.load(f)
        return mean_std

    def run(self, i):
        outputs = []
        smiles = []
        with torch.no_grad():
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
                output = self.n_models[i](**batch['net_input']).logits
                outputs.append(output.cpu().detach().numpy())
                smiles += batch["net_input"]["batched_data"]["smiles"]
        y_pred_logits, smiles = np.concatenate(outputs), np.array(smiles)
        
        for i in range(y_pred_logits.shape[1]):
            task = self.id2task[i]
            if i < self.num_cls:
                y_pred_logits[:, i] = sigmoid(y_pred_logits[:, i])
            else:
                mean = self.mean_std[task]['mean']
                std = self.mean_std[task]['std']
                y_pred_logits[:, i] = y_pred_logits[:, i] * std + mean
        df_pred_score = pd.DataFrame(y_pred_logits, columns=self.task_info[0])
        return smiles, df_pred_score

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def update_args(args, config):
    if config is not None:
        for k, v in config.items():
            args.__dict__[k] = v
    return args
def update_only_args(args, config):
    if config is not None:
        for k in config.keys():
            for k_ , v in config[k].items():
                args.__dict__[k_] = v
    return args
def infer(config=None):
    args_dict = {
        'kk': -1,
        'config': 'task',
        'gpus': '0',
        'nni': 'False'
    }
    args = argparse.Namespace(**args_dict)
    kk = args.kk
    file = open(os.path.join('admet_predictor/configs', f'{args.config}.yml'), 'r', encoding="utf-8")
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    args = update_only_args(args, cfg)
    # args = parse_cfg(parser, cfg)
    args = update_args(args, config)
    args.down_stream_task = 'admet_all'
    args.gumbel_fusion = False
    args.task_name = 'BEMT'
    if not os.path.isfile(args.test_data_path):
        args.test_data_path = os.path.join(args.test_data_path, f'pubchem77M_{kk}.csv')

    Runner = TransformerInference(args, gpu=True)
    df_res = None
    smiles = None
    for i in range(len(Runner.n_models)):
        if df_res is None:
            smiles, df_res = Runner.run(i)
        else:
            _, df = Runner.run(i)
            df_res += df
    df_res /= len(Runner.n_models)
    df_smi = pd.DataFrame(smiles, columns=['smi'])
    df = pd.concat([df_smi, df_res], axis=1)
    return df, kk

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config

def admet_infer():
    from process_pubchem77m_to_csv import txt2csv
    txt2csv()
    config = load_yaml_config('./admet_predictor/configs/configs.yml')
    df_res, kk = infer(config=config)
    df_res.to_csv('admet_predictor/temp/temp_g.csv', index=False)

def admet_infer2():
    from process_pubchem77m_to_csv import txt2csv
    txt2csv()
    config = load_yaml_config('./admet_predictor/configs/configs.yml')
    df_res, kk = infer(config=config)
    return df_res
   
   

def infer2(config=None,random_num=0):
    # args, parser = parse_args()
    args_dict = {
        'kk': -1,
        'config': 'task',
        'gpus': '0',
        'nni': 'False'
    }
    args = argparse.Namespace(**args_dict)
    kk = args.kk

    args.test_data_path='admet_predictor/temps/temp'+ str(random_num)+'.csv'
    file = open(os.path.join('admet_predictor/configs', f'{args.config}.yml'), 'r', encoding="utf-8")
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    args = update_only_args(args, cfg)
    # args = parse_cfg(parser, cfg)
    args = update_args(args, config)
    args.down_stream_task = 'admet_all'
    args.gumbel_fusion = False
    args.task_name = 'BEMT'
    if not os.path.isfile(args.test_data_path):
        args.test_data_path = os.path.join(args.test_data_path, f'pubchem77M_{kk}.csv')

    Runner = TransformerInference(args, gpu=True)
    df_res = None
    smiles = None
    for i in range(len(Runner.n_models)):
        if df_res is None:
            smiles, df_res = Runner.run(i)
        else:
            _, df = Runner.run(i)
            df_res += df
    df_res /= len(Runner.n_models)
    df_smi = pd.DataFrame(smiles, columns=['smi'])
    df = pd.concat([df_smi, df_res], axis=1)
    return df, kk

def admet_infer3(random_num):
    '''
    添加参数
    '''
    from process_pubchem77m_to_csv import txt2csv,txt2csv2
    txt2csv2(random_num)
    config = load_yaml_config('./admet_predictor/configs/configs.yml')
    df_res, kk = infer2(config=config,random_num=random_num)


    return df_res

if __name__ == '__main__':
    from process_pubchem77m_to_csv import txt2csv
    txt2csv()
    config = load_yaml_config('/mnt/e/tangui/SMILES_NEW/admet_predictor/configs/configs.yml')
    df_res, kk = infer(config=config)
    df_res.to_csv('/mnt/e/tangui/SMILES_NEW/admet_predictor/temp/temp_g.csv', index=False)
    # df_res = admet_infer2()

