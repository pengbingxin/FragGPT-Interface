import json
import os
import lmdb
import numpy as np
import pickle

from rdkit import Chem
from rdkit.Chem import BRICS

import pandas as pd
import torch

from tqdm import tqdm

from datasets import register_datasets
from datasets.tokenizer import *
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

classification_names = ['Ames', 'BBB', 'Carcinogenicity', 'CYP1A2-inh', 'CYP1A2-sub', 'CYP2C19-inh', 'CYP2C19-sub',
                            'CYP2C9-inh', 'CYP2C9-sub', 'CYP2D6-inh', 'CYP2D6-sub', 'CYP3A4-inh', 'CYP3A4-sub', 'DILI',
                            'EC', 'EI', 'F(20%)', 'F(50%)', 'FDAMDD', 'hERG', 'H-HT', 'HIA', 'MLM', 'NR-AhR', 'NR-AR',
                            'NR-AR-LBD', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'Pgp-inh', 'Pgp-sub',
                            'Respiratory', 'ROA', 'SkinSen', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53', 'T12']
regression_names = ['BCF', 'Caco-2', 'CL', 'Fu', 'IGC50', 'LC50', 'LC50DM', 'LogD', 'LogP', 'LogS', 'MDCK', 'PPB', 'VDss']

tasks = classification_names + regression_names



@register_datasets(['smiles_admet'])
class SmilesDataset():
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.data_path = cfg.DATA.DATA_ROOT
        # self.data_path = os.path.join(self.data_dir, 'admet_pre10.csv')
        if mode == "train":
            filenames = ['admet_pre10_train.csv', 'pubchem_admet_0.csv', 'pubchem_admet_1.csv', 'pubchem_admet_2.csv',
                    'pubchem_admet_3.csv', 'pubchem_admet_4.csv', 'pubchem_admet_5.csv', 'pubchem_admet_6.csv', 
                    'pubchem_admet_7.csv']
        else:
            filenames = ['admet_pre10_valid.csv']
        self.df = None
        all_names = os.listdir(self.data_path)
        for idx, name in tqdm(enumerate(all_names), total=len(all_names)):
            """
            if name not in  ['admet_pre10_train.csv', 'admet_pre10_valid.csv']:
                continue
            """
            if name.endswith('.csv') and name in filenames:
                fil = os.path.join(self.data_path, name)
                dataframe = pd.read_csv(fil)
                cols = dataframe.columns
                t = {}
                for c in cols:
                    if c == 'smi':
                        t[c] = dataframe[c].dtype
                    else:
                        t[c] = 'float16'
                dataframe = dataframe.astype(t)
                if self.df is None:
                    self.df = dataframe
                else:
                    self.df = pd.concat([self.df, dataframe], axis=0)
        self.df.reset_index().drop(columns='index')
        # self.df = pd.read_csv(self.data_path)
        self.df = self.df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)

        if mode == 'train':
            self.df = self.df.iloc[:int(len(self.df)*0.9), :]
        elif mode == 'valid':
            self.df = self.df.iloc[int(len(self.df)*0.9):, :]

        with open(os.path.join(self.data_path, 'mean_std.json'), 'r') as f:
            self.mean_std = json.load(f)
        self.tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
            cfg.MODEL.TOKENIZER_PATH,
            model_max_length=cfg.DATA.MAX_SMILES_LEN
        )
        self.max_seq_len = cfg.DATA.MAX_SMILES_LEN
        self.end_id = self.tokenizer.convert_tokens_to_ids("</s>")
        self.cls_id = self.tokenizer.convert_tokens_to_ids("<cls>")

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
        return len(self.df)

    def __getitem__(self, item):
        data = self.df.iloc[item, :]
        smiles = data['smi']
        admet_prop = []
        for task in tasks:
            _data = data[task]
            if task in self.mean_std:
                mean = self.mean_std[task]['mean']
                std = self.mean_std[task]['std']
                d = (_data - mean) / std
                admet_prop.append(d)
            else:
                admet_prop.append(_data)
        try:
            fragments = self.get_fragment(smiles)
            fragment = str(np.random.choice(fragments, 1)[0])
        except:
            fragment = smiles
        return {'smiles': smiles, 'admet_prop': admet_prop, 'fragment': fragment}

    def get_fragment(self, smiles):
        clean_fragments = get_frames(smiles)
        clean_fragments = [frag for frag in clean_fragments if len(frag) > 2]
        return clean_fragments

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles'] for data in batch]
        token_batch = self.tokenizer(smiles_batch)
        # token_batch['input_ids'] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        token_batch_ids = []
        for token_id in token_batch['input_ids']:
            if len(token_id) > self.max_seq_len:
                token_id = token_id[:self.max_seq_len]
            elif len(token_id) < self.max_seq_len:
                token_id = token_id +[0]*(self.max_seq_len-len(token_id))
            token_batch_ids.append(token_id)
        token_batch_ids = torch.LongTensor(token_batch_ids)

        labels = token_batch_ids.clone()
        labels[labels == 0] = -100
        # labels[(labels[:, -1] != -100) & (labels[:, -1] != self.end_id), -1] = self.end_id
        new_batch["labels"] = labels
        # add cls token
        new_batch['smiles_ids'] = torch.cat([token_batch_ids, self.cls_id * torch.ones((token_batch_ids.shape[0], 1))],
                                             dim=1).long()

        fragment_batch = [data['fragment'] for data in batch]
        frag_batch = self.tokenizer(fragment_batch)
        frag_batch['input_ids'] = self._torch_collate_batch(frag_batch['input_ids'], self.tokenizer)
        new_batch['fragment_ids'] = frag_batch['input_ids']

        admet_prop_batch = [data['admet_prop'] for data in batch]
        admet_prop_batch = torch.tensor(admet_prop_batch)
        new_batch['admet_prop'] = admet_prop_batch

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


def fragment_recursive(mol, frags):
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))
        if len(bonds) == 0:
            frags.append(Chem.MolToSmiles(mol))
            return frags

        idxs, labs = list(zip(*bonds))
        # print(bonds)
        # print(idxs, labs)
        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())
        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]
        broken = Chem.FragmentOnBonds(mol,
                                      bondIndices=[bond_idxs[0]],
                                      dummyLabels=[(0, 0)])
        head, tail = Chem.GetMolFrags(broken, asMols=True)
        # print(mol_to_smiles(head), mol_to_smiles(tail))
        frags.append(Chem.MolToSmiles(head))
        return fragment_recursive(tail, frags)
    except Exception as e:
        #         print (e)
        pass


def remove_dummy(smiles):
    try:
        stripped_smi = smiles.replace('*', '[H]')
        mol = Chem.MolFromSmiles(stripped_smi)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        #         print (e)
        return None


def get_frames(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fragments = fragment_recursive(mol, [])
    if fragments is not None:
        clean_fragments = [remove_dummy(smi) for smi in fragments]
    else:
        clean_fragments = [smiles]
    return clean_fragments
