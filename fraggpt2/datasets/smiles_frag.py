import os
import re
import json
import lmdb
import pickle
import torch
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/chenyu/proj/SMILES_NEW')
from datasets import register_datasets
from rdkit import Chem
from rdkit.Chem import Descriptors
from datasets.tokenizer import *
from datasets.pocket import pdb_to_pocket_data, pdb_to_pocket_data2
from typing import Optional
from utils.fragment import BRICS_RING_R_Fragmenizer, find_all_idx
from utils.fragment2 import fragment as fgt
PATTERN = r'\[\*:(\d+)\]'  # 匹配[*:数字]模式的正则表达式
replacement = r'[\1*]'  # 替换为[数字*]模式的字符串
mapping = {'PAD': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4,
           'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
           'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
           'Y': 19, 'V': 20}

classification_names = ['Ames', 'BBB', 'Carcinogenicity', 'CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh', 'CYP2C19-sub',
                            'CYP2C9-inh', 'CYP2C9-sub', 'CYP2D6-inh', 'CYP2D6-sub', 'CYP3A4-inh', 'CYP3A4-sub', 'DILI',
                            'EC', 'EI', 'F(20%)', 'F(50%)', 'FDAMDD', 'hERG', 'H-HT', 'HIA', 'MLM', 'NR-AhR', 'NR-AR',
                            'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'Pgp-inh', 'Pgp-sub',
                            'Respiratory', 'ROA', 'SkinSen', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'T12']
regression_names = ['BCF', 'Caco-2', 'CL', 'Fu', 'IGC50', 'LC50', 'LC50DM', 'LogD', 'LogP', 'LogS', 'MDCK', 'PPB', 'VDss', 'MW', 'TPSA'] # , 'MW', 'TPSA'
drop_names = ['Carcinogenicity', 'CYP1A2-sub', 'CYP2C9-sub', 'CYP2D6-sub', 'CYP3A4-sub', 'F(20%)', 'F(50%)', 'FDAMDD', 'NR-Aromatase', 'NR-ER', 'NR-PPAR-gamma', 'Pgp-sub', 'Respiratory', 'ROA', 'SkinSen', 'T12']

tasks = classification_names + regression_names
id2task = {i: task for i, task in enumerate(tasks)}
task2id = {task: i for i, task in enumerate(tasks)}

def sub_point_id(string, ts_dict):
    copy_string = string
    last_left = 0
    i = 0
    time = 0
    while i<len(string):
        time += 1
        if time > 2000:
            return copy_string
        s = string[i]
        if s == '[':
            last_left = i
        if s == '*':
            if i+1 < len(string) and string[i+1] == ']':
                point_id = string[last_left+1: i]
                obj_point_id = ts_dict[point_id]
                #string[last_left+1: i] = obj_point_id
                if len(point_id) == 1 and len(obj_point_id) == 1:
                    string = string[:last_left+1] + obj_point_id + string[i:]
                elif len(point_id) == 2 and len(obj_point_id) == 1:
                    string = string[:last_left+1] + obj_point_id + string[i:]
                    i -= 1
                elif len(point_id) == 1 and len(obj_point_id) == 2:
                    string = string[:last_left+1] + obj_point_id + string[i:]
                    i += 1
                elif len(point_id) == 2 and len(obj_point_id) == 2:
                    string = string[:last_left+1] + obj_point_id + string[i:]
        i += 1
    return string

@register_datasets(['smiles_frag'])
class FragSmilesDataset():
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.data_dir = cfg.DATA.DATA_ROOT
        self.data_path = os.path.join(self.data_dir, f'{mode}.txt')

        self.smiles_data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for smi in f.readlines():
                self.smiles_data.append(smi.strip())
        self.max_seq_len = cfg.DATA.MAX_SMILES_LEN
        self.fragmenizer = BRICS_RING_R_Fragmenizer(break_ring=False)

        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_SMILES_LEN
        )
        self.end_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return len(self.smiles_data)

    def get_frag_smiles(self, smiles):
        if self.cfg.fragment2 :
            fragment_list = fgt(smiles)
            idx = np.random.choice(len(fragment_list), 1)[0]
            res = fragment_list[idx].split('.')
        else:
            mol = Chem.MolFromSmiles(smiles)
            frags, _ = self.fragmenizer.fragmenize(mol)
            frags = Chem.GetMolFrags(frags, asMols=True)
            res = [Chem.MolToSmiles(frag) for frag in frags]
        return res

    def smiles_augment(self, smiles):
        new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True, isomericSmiles=True)
        frags_list = self.get_frag_smiles(new_smiles)
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
        # np.random.seed(42)
        np.random.shuffle(ts)
        new_frags_list = []
        ts_dict = {str(j): str(i + 1) for i, j in enumerate(ts)}
        for frag in frags_list:
            new_frags_list.append(sub_point_id(frag, ts_dict))
        return new_frags_list

    def smiles_augment_frag(self, smiles):
        frag = smiles.split(' ')[1]
        frag = re.sub(PATTERN, replacement, frag)
        frags_list = frag.split('.')
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
        # np.random.seed(42)
        np.random.shuffle(ts)
        new_frags_list = []
        ts_dict = {str(j): str(i + 1) for i, j in enumerate(ts)}
        for frag in frags_list:
            new_frags_list.append(sub_point_id(frag, ts_dict))
        return new_frags_list

    def last_idx(self, all_frags):
        for i, frag in enumerate(all_frags):
            if np.random.rand(1) > 0.5:
                last_all_idx = find_all_idx([frag])
                np.random.shuffle(last_all_idx)
                rand_idx = last_all_idx[0]
                time = 0
                new_last_frag = None
                while time < 10000:
                    mol = Chem.MolFromSmiles(frag)
                    try:
                        new_last_frag = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
                    except:
                        new_last_frag = frag
                        break
                    if new_last_frag.endswith(f'[{rand_idx}*]'):
                        break
                    time += 1
                if new_last_frag is None:
                    new_last_frag = frag
                all_frags[i] = new_last_frag
        return all_frags

    def __getitem__(self, item):
        data = self.smiles_data[item]
        # if '.' in data:
        #     data = sorted(data.split('.'), key=lambda x: len(x), reverse=True)[0]
        try:
            if '*' in data:
                all_frags = self.smiles_augment_frag(data)
            else:
                all_frags = self.smiles_augment(data)
        except:
            all_frags = self.smiles_augment(self.smiles_data[0])
        """
        if float(np.random.rand(1)[0]) > 0.5:
            all_frags = self.last_idx(all_frags)
        """
        # np.random.seed(self.cfg.seed)
        np.random.shuffle(all_frags)
        smiles_frag = '<sep>'.join(all_frags)
        return {'smiles_frag': smiles_frag}

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles_frag'] for data in batch]
        token_batch = self.tokenizer(smiles_batch, truncation=True)
        new_batch["input_ids"] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        labels = new_batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        new_batch["labels"] = labels
        return new_batch

    def _torch_collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

@register_datasets(['smiles_frag_admet'])
class FragSmilesADMETDataset():
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.data_dir = cfg.DATA.DATA_ROOT
        # self.data_path = os.path.join(self.data_dir, f'admet_11w_{mode}.lmdb')
        # self.data_path = os.path.join(self.data_dir, f'preprocess_{mode}.lmdb')
        self.data_path = os.path.join(self.data_dir, f'{mode}.lmdb')
        self.env, self.txn = self.read_lmdb(self.data_path)
        # self.keys = list(self.txn.cursor().iternext(values=False))
        
        # if mode == 'train':
        #     self.num_data = 77407713
        # else:
        #     self.num_data = 19999

        _keys = list(self.txn.cursor().iternext(values=False))
        self.num_data = len(_keys)

        self.radius = 20
        self.num_types = len(mapping)
        self.max_seq_len = cfg.DATA.MAX_SMILES_LEN
        self.fragmenizer = BRICS_RING_R_Fragmenizer(break_ring=False)

        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_SMILES_LEN
        )
        self.end_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")

        with open(os.path.join(self.data_dir, 'mean_std.json'), 'r') as f:
            self.mean_std = json.load(f)

    def read_lmdb(self, lmdb_path):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        return env, txn

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return self.num_data

    def get_frag_smiles(self, smiles):
        if self.cfg.fragment2 :
            fragment_list = fgt(smiles)
            idx = np.random.choice(len(fragment_list), 1)[0]
            res = fragment_list[idx].split('.')
        else:
            mol = Chem.MolFromSmiles(smiles)
            frags, _ = self.fragmenizer.fragmenize(mol)
            frags = Chem.GetMolFrags(frags, asMols=True)
            res = [Chem.MolToSmiles(frag) for frag in frags]
        return res

    def smiles_augment_frag(self, smiles):
        frag = smiles.split(' ')[1]
        # frag = frag[0] + '.' + frag[1]
        frag = re.sub(PATTERN, replacement, frag)
        frags_list = frag.split('.')
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
        # np.random.seed(42)
        np.random.shuffle(ts)
        new_frags_list = []
        ts_dict = {str(j): str(i + 1) for i, j in enumerate(ts)}
        for frag in frags_list:
            new_frags_list.append(sub_point_id(frag, ts_dict))
        return new_frags_list
    def smiles_augment(self, smiles):
        try:
            new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True, isomericSmiles=True)
        except:
            new_smiles = smiles
        frags_list = self.get_frag_smiles(new_smiles)
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
        # np.random.seed(42)
        np.random.shuffle(ts)
        new_frags_list = []
        ts_dict = {str(j): str(i + 1) for i, j in enumerate(ts)}
        for frag in frags_list:
            new_frags_list.append(sub_point_id(frag, ts_dict))
        return new_frags_list

    def last_idx(self, all_frags):
        for i, frag in enumerate(all_frags):
            if np.random.rand(1) > 0.5:
                last_all_idx = find_all_idx([frag])
                np.random.shuffle(last_all_idx)
                rand_idx = last_all_idx[0]
                time = 0
                new_last_frag = None
                while time < 10000:
                    mol = Chem.MolFromSmiles(frag)
                    try:
                        new_last_frag = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
                    except:
                        new_last_frag = frag
                        break
                    if new_last_frag.endswith(f'[{rand_idx}*]'):
                        break
                    time += 1
                if new_last_frag is None:
                    new_last_frag = frag
                all_frags[i] = new_last_frag
        return all_frags

    def __getitem__(self, item):
        # datapoint_pickled = self.txn.get(self.keys[item])
        datapoint_pickled = self.txn.get(str(item).encode())
        # try:
        data = pickle.loads(datapoint_pickled)['infos']
        # except:
        #     print(datapoint_pickled, item)
        #     datapoint_pickled = self.txn.get(str(0).encode())
        #     data = pickle.loads(datapoint_pickled)['infos']
        smiles = data['smiles']
        # if '.' in smiles:
        #     smiles = sorted(smiles.split('.'), key=lambda x: len(x), reverse=True)[0]

        try:
            if '*' in data:
                all_frags = self.smiles_augment_frag(smiles)
            else:
                all_frags = self.smiles_augment(smiles)

            #all_frags = self.smiles_augment(smiles)
            # all_frags = self.smiles_augment_frag(smiles)
            admet = data['admet_prop']
        except:
            datapoint_pickled = self.txn.get(str(0).encode())
            data = pickle.loads(datapoint_pickled)['infos']
            smiles = data['smiles']
            print('222222222', smiles)
            all_frags = self.smiles_augment_frag(smiles)
            # all_frags = self.smiles_augment(smiles)
            admet = data['admet_prop']
            
        admet_prop = []
        for k in range(admet.shape[0]):
            task = id2task[k]
            if task in drop_names:
                continue
            if task in self.mean_std:
                mean = self.mean_std[task]['mean']
                std = self.mean_std[task]['std']
                admet_prop.append(float((admet[k] - mean) / std))
            else:
                admet_prop.append(float(admet[k]))

        if len(admet_prop) == 38:
            mw = Descriptors.MolWt(Chem.MolFromSmiles(smiles))
            mw = (mw - self.mean_std['MW']['mean']) / self.mean_std['MW']['std']
            admet_prop.append(mw)

            tpsa = Descriptors.TPSA(Chem.MolFromSmiles(smiles))
            tpsa = (tpsa - self.mean_std['TPSA']['mean']) / self.mean_std['TPSA']['std']
            admet_prop.append(tpsa)


        assert len(admet_prop) == len(tasks) - len(drop_names)
        # np.random.seed(self.cfg.seed)
        np.random.shuffle(all_frags)
        smiles_frag = '<sep>'.join(all_frags)
        return {'smiles_frag': smiles_frag, 'admet': admet_prop}

    def get_distance_matrix(self, pos):
        assert pos.shape[1] == 3, 'The shape of pos is error! '
        return torch.pow((pos.unsqueeze(1) - pos.unsqueeze(0)), 2).sum(-1) ** 0.5

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles_frag'] for data in batch]
        token_batch = self.tokenizer(smiles_batch, truncation=True)
        new_batch["input_ids"] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        labels = new_batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        new_batch["labels"] = labels
        new_batch['admet'] = torch.tensor([data['admet'] for data in batch])
        return new_batch

    def _torch_collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result


@register_datasets(['target'])
class FragSmilesPocketTargetDataset():
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.data_dir = cfg.DATA.DATA_ROOT
        self.df = pd.read_csv(self.data_dir, encoding='gbk')
        self.smiles = self.df['Smiles'].tolist()
        self.smiles = [smi for smi in self.smiles if Chem.MolFromSmiles(smi) is not None]
        # np.random.seed(cfg.seed)
        np.random.shuffle(self.smiles)
        if mode == 'train':
            self.smiles = self.smiles[:int(0.9*len(self.smiles))]
        else:
            self.smiles = self.smiles[int(0.9 * len(self.smiles)):]

        self.max_res_seq_len = cfg.DATA.MAX_RES_LEN
        self.radius = 20.0
        self.num_types = len(mapping)
        self.max_seq_len = cfg.DATA.MAX_SMILES_LEN
        self.fragmenizer = BRICS_RING_R_Fragmenizer()

        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_SMILES_LEN
        )
        self.end_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return len(self.smiles)

    def get_frag_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        frags, _ = self.fragmenizer.fragmenize(mol)
        frags = Chem.GetMolFrags(frags, asMols=True)
        res = [Chem.MolToSmiles(frag) for frag in frags]
        return res

    def smiles_augment(self, smiles):
        try:
            new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True, isomericSmiles=True)
        except:
            new_smiles = smiles
        frags_list = self.get_frag_smiles(new_smiles)
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
        # np.random.seed(42)
        np.random.shuffle(ts)
        new_frags_list = []
        ts_dict = {str(j): str(i + 1) for i, j in enumerate(ts)}
        for frag in frags_list:
            new_frags_list.append(sub_point_id(frag, ts_dict))
        return new_frags_list

    def last_idx(self, all_frags):
        for i, frag in enumerate(all_frags):
            if np.random.rand(1) > 0.5:
                last_all_idx = find_all_idx([frag])
                np.random.shuffle(last_all_idx)
                rand_idx = last_all_idx[0]
                time = 0
                new_last_frag = None
                while time < 10000:
                    mol = Chem.MolFromSmiles(frag)
                    try:
                        new_last_frag = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
                    except:
                        new_last_frag = frag
                        break
                    if new_last_frag.endswith(f'[{rand_idx}*]'):
                        break
                    time += 1
                if new_last_frag is None:
                    new_last_frag = frag
                all_frags[i] = new_last_frag
        return all_frags

    def __getitem__(self, item):
        ref_smiles, protein_dict = pdb_to_pocket_data(pdb_file=os.path.join(self.cfg.DATA.POCKET, 'pocket.pdb'),
                                                      bbox_size=self.radius,
                                                      mol_file=os.path.join(self.cfg.DATA.POCKET, 'ligand.sdf'),
                                                      perturb=True)

        smiles = self.smiles[item]

        res_seq = protein_dict['res_seq']
        res_seq = torch.LongTensor([mapping[i] for i in res_seq])
        res_edge_type = res_seq.view(-1, 1) * self.num_types + res_seq.view(1, -1)
        res_dis = self.get_distance_matrix(protein_dict['pkt_node_xyz'])
        res_coords = protein_dict['pkt_node_xyz']

        if '.' in smiles:
            smiles = sorted(smiles.split('.'), key=lambda x: len(x), reverse=True)[0]
        all_frags = self.smiles_augment(smiles)
        # np.random.seed(self.cfg.seed)
        np.random.shuffle(all_frags)
        smiles_frag = '<sep>'.join(all_frags)
        return {'smiles_frag': smiles_frag, 'res_seq': res_seq, 'res_dis': res_dis,
                'res_coords': res_coords, 'res_edge_type': res_edge_type}

    def get_distance_matrix(self, pos):
        assert pos.shape[1] == 3, 'The shape of pos is error! '
        return torch.pow((pos.unsqueeze(1) - pos.unsqueeze(0)), 2).sum(-1) ** 0.5

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles_frag'] for data in batch]
        token_batch = self.tokenizer(smiles_batch, truncation=True)
        new_batch["input_ids"] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        labels = new_batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        new_batch["labels"] = labels

        pad_pocket_seq = []
        pad_pocket_edge_type = []
        pad_pocket_dis = []
        max_pocket_length = min(self.max_res_seq_len, max([d['res_seq'].shape[0] for d in batch]))
        for d in batch:
            pad_pocket_seq.append(pad_to_max_length_1d(d['res_seq'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_edge_type.append(pad_to_max_length_2d(d['res_edge_type'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_dis.append(pad_to_max_length_2d(d['res_dis'], max_pocket_length).unsqueeze(0))

        new_batch['pocket_seq'] = torch.cat(pad_pocket_seq)
        new_batch['pocket_edge_type'] = torch.cat(pad_pocket_edge_type)
        new_batch['pocket_dis'] = torch.cat(pad_pocket_dis)
        return new_batch

    def _torch_collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

@register_datasets(['smiles_frag_pocket'])
class FragSmilesPocketDataset():
    def __init__(self, cfg, mode='train'):
        # mode = 'valid'
        self.cfg = cfg
        self.data_dir = cfg.DATA.DATA_ROOT
        """
        if self.cfg.MODEL.ADMET_ENCODER.use_admet:
            self.data_path = os.path.join(self.data_dir, f'process_{mode}_admet.lmdb')
        else:
            self.data_path = os.path.join(self.data_dir, f'process_{mode}.lmdb')
        """
        self.data_path = os.path.join(self.data_dir, f'process_{mode}_admet.lmdb')
        self.env, self.txn = self.read_lmdb(self.data_path)
        self.keys = list(self.txn.cursor().iternext(values=False))

        self.max_res_seq_len = cfg.DATA.MAX_RES_LEN
        self.radius = 20
        self.num_types = len(mapping)
        self.max_seq_len = cfg.DATA.MAX_SMILES_LEN
        self.fragmenizer = BRICS_RING_R_Fragmenizer(break_ring=False)

        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_SMILES_LEN
        )
        self.end_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")

        with open(os.path.join(self.data_dir, 'mean_std.json'), 'r') as f:
            self.mean_std = json.load(f)

    def read_lmdb(self, lmdb_path):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        return env, txn

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return len(self.keys)

    def get_frag_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        frags, _ = self.fragmenizer.fragmenize(mol)
        frags = Chem.GetMolFrags(frags, asMols=True)
        res = [Chem.MolToSmiles(frag) for frag in frags]
        return res

    def smiles_augment(self, smiles):
        try:
            new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True, isomericSmiles=True)
        except:
            new_smiles = smiles
        frags_list = self.get_frag_smiles(new_smiles)
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
        # np.random.seed(42)
        np.random.shuffle(ts)
        new_frags_list = []
        ts_dict = {str(j): str(i + 1) for i, j in enumerate(ts)}
        for frag in frags_list:
            new_frags_list.append(sub_point_id(frag, ts_dict))
        return new_frags_list

    def last_idx(self, all_frags):
        for i, frag in enumerate(all_frags):
            if np.random.rand(1) > 0.5:
                last_all_idx = find_all_idx([frag])
                np.random.shuffle(last_all_idx)
                rand_idx = last_all_idx[0]
                time = 0
                new_last_frag = None
                while time < 10000:
                    mol = Chem.MolFromSmiles(frag)
                    try:
                        new_last_frag = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
                    except:
                        new_last_frag = frag
                        break
                    if new_last_frag.endswith(f'[{rand_idx}*]'):
                        break
                    time += 1
                if new_last_frag is None:
                    new_last_frag = frag
                all_frags[i] = new_last_frag
        return all_frags

    def __getitem__(self, item):
        datapoint_pickled = self.txn.get(self.keys[item])
        data = pickle.loads(datapoint_pickled)
        smiles = data['smiles']
        pockets = data['pocket']
        admet = data['admet']
        admet_prop = []
        for k in range(admet.shape[0]):
            task = id2task[k]
            if task in drop_names:
                continue
            if task in self.mean_std:
                mean = self.mean_std[task]['mean']
                std = self.mean_std[task]['std']
                admet_prop.append(float((admet[k] - mean) / std))
            else:
                admet_prop.append(float(admet[k]))

        random_idx = int(np.random.choice(len(pockets), 1)[0])
        protein_dict = pockets[random_idx]

        res_seq = protein_dict['res_seq']
        res_seq = torch.LongTensor([mapping[i] for i in res_seq])
        res_edge_type = res_seq.view(-1, 1) * self.num_types + res_seq.view(1, -1)
        res_dis = self.get_distance_matrix(protein_dict['pkt_node_xyz'])
        res_coords = protein_dict['pkt_node_xyz']

        if '.' in smiles:
            smiles = sorted(smiles.split('.'), key=lambda x: len(x), reverse=True)[0]
        while 1:
            try:
                all_frags = self.smiles_augment(smiles)
                break
            except:
                print(smiles)
        """
        if float(np.random.rand(1)[0]) > 0.5:
            all_frags = self.last_idx(all_frags)
        """
        # np.random.seed(self.cfg.seed)
        np.random.shuffle(all_frags)
        smiles_frag = '<sep>'.join(all_frags)
        return {'smiles_frag': smiles_frag, 'res_seq': res_seq, 'res_dis': res_dis,
                'res_coords': res_coords, 'res_edge_type': res_edge_type, 'admet': admet_prop}

    def get_distance_matrix(self, pos):
        assert pos.shape[1] == 3, 'The shape of pos is error! '
        return torch.pow((pos.unsqueeze(1) - pos.unsqueeze(0)), 2).sum(-1) ** 0.5

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles_frag'] for data in batch]
        token_batch = self.tokenizer(smiles_batch, truncation=True)
        new_batch["input_ids"] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        labels = new_batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        new_batch["labels"] = labels

        pad_pocket_seq = []
        pad_pocket_edge_type = []
        pad_pocket_dis = []
        max_pocket_length = min(self.max_res_seq_len, max([d['res_seq'].shape[0] for d in batch]))
        for d in batch:
            pad_pocket_seq.append(pad_to_max_length_1d(d['res_seq'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_edge_type.append(pad_to_max_length_2d(d['res_edge_type'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_dis.append(pad_to_max_length_2d(d['res_dis'], max_pocket_length).unsqueeze(0))

        new_batch['pocket_seq'] = torch.cat(pad_pocket_seq)
        new_batch['pocket_edge_type'] = torch.cat(pad_pocket_edge_type)
        new_batch['pocket_dis'] = torch.cat(pad_pocket_dis)
        new_batch['admet'] = torch.tensor([data['admet'] for data in batch])
        return new_batch

    def _torch_collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

from tqdm import tqdm
# @register_datasets(['smiles_frag_pocket_pretrain'])
class FragSmilesPocketPretrainDataset():
    def __init__(self, cfg, mode='train'):
        mode = 'valid'
        self.cfg = cfg
        self.data_dir = os.path.join(cfg.DATA.DATA_ROOT, mode + '.json')
        self.pdb_path = cfg.DATA.PDB_DATA_PATH
        self.data = []
        print('加载数据')
        with open(self.data_dir, 'r') as f:
            infos = json.load(f)
        for info in tqdm(infos, total=len(infos)):
            gscore = info[0]
            pkt_id = info[1]
            lig_path = info[3]
            index = info[4]
            pkt_path = os.path.join(self.pdb_path, pkt_id, f'{pkt_id}_protein_processed.pdb')
            # if os.path.exists(pkt_path):
            self.data.append([pkt_path, lig_path, index])

        self.max_res_seq_len = cfg.DATA.MAX_RES_LEN
        self.radius = 20
        self.num_types = len(mapping)
        self.max_seq_len = cfg.DATA.MAX_SMILES_LEN
        self.fragmenizer = BRICS_RING_R_Fragmenizer()

        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_SMILES_LEN
        )
        self.end_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")
        self.pad_id = self.tokenizer.convert_tokens_to_ids("<pad>")

    def read_lmdb(self, lmdb_path):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        txn = env.begin()
        return env, txn

    @classmethod
    def build_datasets(cls, cfg, mode):
        return cls(cfg, mode)

    def __len__(self):
        return len(self.data)

    def get_frag_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        frags, _ = self.fragmenizer.fragmenize(mol)
        frags = Chem.GetMolFrags(frags, asMols=True)
        res = [Chem.MolToSmiles(frag) for frag in frags]
        return res

    def smiles_augment(self, smiles):
        try:
            new_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), doRandom=True, isomericSmiles=True)
        except:
            new_smiles = smiles
        frags_list = self.get_frag_smiles(new_smiles)
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
        # np.random.seed(42)
        np.random.shuffle(ts)
        new_frags_list = []
        ts_dict = {str(j): str(i + 1) for i, j in enumerate(ts)}
        for frag in frags_list:
            new_frags_list.append(sub_point_id(frag, ts_dict))
        return new_frags_list

    def last_idx(self, all_frags):
        for i, frag in enumerate(all_frags):
            if np.random.rand(1) > 0.5:
                last_all_idx = find_all_idx([frag])
                np.random.shuffle(last_all_idx)
                rand_idx = last_all_idx[0]
                time = 0
                new_last_frag = None
                while time < 10000:
                    mol = Chem.MolFromSmiles(frag)
                    try:
                        new_last_frag = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
                    except:
                        new_last_frag = frag
                        break
                    if new_last_frag.endswith(f'[{rand_idx}*]'):
                        break
                    time += 1
                if new_last_frag is None:
                    new_last_frag = frag
                all_frags[i] = new_last_frag
        return all_frags

    def __getitem__(self, item):
        data = self.data[item]
        smiles, protein_dict = pdb_to_pocket_data2(data[0], self.radius, data[1], id=data[2])
        res_seq = protein_dict['res_seq']
        res_seq = torch.LongTensor([mapping[i] for i in res_seq])
        res_edge_type = res_seq.view(-1, 1) * self.num_types + res_seq.view(1, -1)
        res_dis = self.get_distance_matrix(protein_dict['pkt_node_xyz'])
        res_coords = protein_dict['pkt_node_xyz']

        if '.' in smiles:
            smiles = sorted(smiles.split('.'), key=lambda x: len(x), reverse=True)[0]
        while 1:
            try:
                all_frags = self.smiles_augment(smiles)
                break
            except:
                print(smiles)

        if float(np.random.rand(1)[0]) > 0.5:
            all_frags = self.last_idx(all_frags)
        # np.random.seed(self.cfg.seed)
        np.random.shuffle(all_frags)
        smiles_frag = '<sep>'.join(all_frags)
        return {'smiles_frag': smiles_frag, 'res_seq': res_seq, 'res_dis': res_dis,
                'res_coords': res_coords, 'res_edge_type': res_edge_type}

    def get_distance_matrix(self, pos):
        assert pos.shape[1] == 3, 'The shape of pos is error! '
        return torch.pow((pos.unsqueeze(1) - pos.unsqueeze(0)), 2).sum(-1) ** 0.5

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles_frag'] for data in batch]
        token_batch = self.tokenizer(smiles_batch, truncation=True)
        new_batch["input_ids"] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        labels = new_batch["input_ids"].clone()
        labels[labels == self.pad_id] = -100
        new_batch["labels"] = labels

        pad_pocket_seq = []
        pad_pocket_edge_type = []
        pad_pocket_dis = []
        max_pocket_length = min(self.max_res_seq_len, max([d['res_seq'].shape[0] for d in batch]))
        for d in batch:
            pad_pocket_seq.append(pad_to_max_length_1d(d['res_seq'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_edge_type.append(pad_to_max_length_2d(d['res_edge_type'], max_pocket_length).type(torch.LongTensor).unsqueeze(0))
            pad_pocket_dis.append(pad_to_max_length_2d(d['res_dis'], max_pocket_length).unsqueeze(0))

        new_batch['pocket_seq'] = torch.cat(pad_pocket_seq)
        new_batch['pocket_edge_type'] = torch.cat(pad_pocket_edge_type)
        new_batch['pocket_dis'] = torch.cat(pad_pocket_dis)
        return new_batch

    def _torch_collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        import torch

        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple, np.ndarray)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        length_of_first = examples[0].size(0)

        # Check if padding is necessary.

        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

def pad_to_max_length_1d(x, max_length):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    l = x.shape[0]
    new_x = torch.zeros_like(x)
    if l > max_length:
        new_x = new_x[:max_length]
    else:
        x_shape = [i for i in x.shape]
        x_shape[0] = max_length - l
        new_x = torch.cat([new_x, torch.zeros(tuple(x_shape))])
    new_x[:l] = x[:max_length]
    return new_x

def pad_to_max_length_2d(x, max_length):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    l = x.shape[0]
    new_x = torch.zeros((max_length, max_length))
    new_x[:l, :l] = x[:max_length, :max_length]
    return new_x

if __name__ == '__main__':
    smiles = 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
    mol = Chem.MolFromSmiles(smiles)
    fragmenizer = BRICS_RING_R_Fragmenizer(break_ring=False)
    frags, _ = fragmenizer.fragmenize(mol)
    frags = Chem.GetMolFrags(frags, asMols=True)
    res = [Chem.MolToSmiles(frag) for frag in frags]
    print(res)
    '''
    O=C(C[*:2])N[*:1] CC(C)(C)c1ccc2occ([*:2])c2c1.Fc1ccccc1[*:1]
    '[*:2]C(C)(C)C.[*:1]c1coc2ccc([*:2])cc12.[*:1]CC([*:3])=O.[*:3]N[*:4].[*:4]c1ccccc1F'
    '[2*]C(C)(C)C.[1*]c1coc2ccc([2*])cc12.[1*]CC([3*])=O.[3*]N[4*].[4*]c1ccccc1F'
    '''
    import re
    pattern = r'\[\*:(\d+)\]'
    replacement = r'[\1*]'
    result = re.sub(pattern, replacement, ' O=C(C[*:6])N[*:1] CC(C)(C)c1ccc2occ([*:2])c2c1.Fc1ccccc1[*:1]')
    print('O=C(C[*:6])N[*:1] CC(C)(C)c1ccc2occ([*:2])c2c1.Fc1ccccc1[*:1]')
    print(result)

