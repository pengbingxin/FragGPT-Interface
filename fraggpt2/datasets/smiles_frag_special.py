import os
import re
import json
import lmdb
import pickle
import torch
import numpy as np
import sys
sys.path.append('/home/chenyu/proj/SMILES_NEW')
from datasets import register_datasets
from rdkit import Chem
from rdkit.Chem import Descriptors
from datasets.tokenizer import *
from datasets.pocket import pdb_to_pocket_data, pdb_to_pocket_data2
from typing import Optional
from utils.fragment import BRICS_RING_R_Fragmenizer, find_all_idx

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


@register_datasets(['smiles_frag_special'])
class FragSmilesSpecialADMETDataset():
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.data_dir = cfg.DATA.DATA_ROOT
        self.data_path = os.path.join(self.data_dir, f'{mode}.lmdb')
        self.env, self.txn = self.read_lmdb(self.data_path)

        _keys = list(self.txn.cursor().iternext(values=False))
        self.num_data = len(_keys)

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
        mol = Chem.MolFromSmiles(smiles)
        frags, _ = self.fragmenizer.fragmenize(mol)
        frags = Chem.GetMolFrags(frags, asMols=True)
        res = [Chem.MolToSmiles(frag) for frag in frags]
        return res

    def breakpoint_augment(self, frags_list):
        pattern = r'\[(\d+)\*?\]'
        ts = set()
        for s in frags_list:
            nums = [int(x) for x in re.findall(pattern, s)]
            for n in nums:
                ts.add(n)
        ts = list(ts)
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
        datapoint_pickled = self.txn.get(str(item).encode())
        data = pickle.loads(datapoint_pickled)['infos']
        smiles = data['smiles']
        linker_frags = data['linker_frags']
        linker = linker_frags['linker']
        frags = linker_frags['frags']
        frag_list = [*frags, linker]
        all_frags = self.breakpoint_augment(frag_list)
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

        if np.random.rand(1)[0] > 0.5:
            all_frags[0], all_frags[1] = all_frags[1], all_frags[0]
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
                result[i, -example.shape[0]:] = example
        return result