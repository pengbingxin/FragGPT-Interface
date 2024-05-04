import os
import re
import numpy as np
import torch
from datasets import register_datasets
from rdkit import Chem
from datasets.tokenizer import *
from typing import Optional
from utils.fragment import BRICS_RING_R_Fragmenizer, find_all_idx

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

@register_datasets(['smiles_fragv2'])
class FragSmilesDatasetv2():
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.data_dir = cfg.DATA.DATA_ROOT
        self.data_path = os.path.join(self.data_dir, f'{mode}.txt')
        self.smiles_data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for smi in f.readlines():
                self.smiles_data.append(smi.strip())
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
        return len(self.smiles_data)

    def get_frag_smiles(self, smiles):
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
        np.random.seed(42)
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
        if '.' in data:
            data = sorted(data.split('.'), key=lambda x: len(x), reverse=True)[0]
        all_frags = self.smiles_augment(data)
        if float(np.random.rand(1)[0]) > 0.5:
            all_frags = self.last_idx(all_frags)
        np.random.seed(self.cfg.seed)
        np.random.shuffle(all_frags)
        smiles_frag = '<sep>'.join(all_frags)
        return {'smiles_frag': smiles_frag}

    def collator(self, batch):
        new_batch = {}
        smiles_batch = [data['smiles_frag'] for data in batch]
        token_batch = self.tokenizer(smiles_batch, truncation=True)
        # token_batch['input_ids'] = self._torch_collate_batch(token_batch['input_ids'], self.tokenizer)
        token_batch_ids = []
        for token_id in token_batch['input_ids']:
            if len(token_id) > self.max_seq_len:
                token_id = token_id[:self.max_seq_len]
            elif len(token_id) < self.max_seq_len:
                token_id = token_id + [0] * (self.max_seq_len - len(token_id))
            token_batch_ids.append(token_id)
        token_batch_ids = torch.LongTensor(token_batch_ids)

        labels = token_batch_ids.clone()
        labels[labels == 0] = -100
        # labels[(labels[:, -1] != -100) & (labels[:, -1] != self.end_id), -1] = self.end_id
        new_batch["labels"] = labels
        # add cls token
        new_batch['input_ids'] = torch.cat([token_batch_ids, self.cls_id * torch.ones((token_batch_ids.shape[0], 1))],
                                            dim=1).long()

        assert new_batch['input_ids'].shape[1] == (self.max_seq_len + 1)
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