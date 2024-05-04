import warnings
from rdkit import RDLogger
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import os
import time
import torch
import argparse
import re
import json
import signal
import numpy as np
import transformers
from utils.utils import args_parse
from datasets.tokenizer import *
from models.gpt_frag import FragSmilesPocketGPT, FragSmilesGPT
from tqdm import trange
from utils.fragment import BRICS_RING_R_Fragmenizer, find_all_idx, reconstruct_mol, combine_all_fragmens, \
    find_all_idx2, conect_all_fragmens
from utils.metrics import *
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import Draw
import seaborn as sns
import torch.nn.functional as F
import torch.distributions as D
from transformers.pytorch_utils import Conv1D
from datasets.pocket import pdb_to_pocket_data
if transformers.__version__.startswith('4.33'):
    from peft import LoraConfig, TaskType, get_peft_model
if transformers.__version__.startswith('4.26'):
    from transformers import GPT2Config, PfeifferConfig
from admet_predictor.inference_self import admet_infer2
from datasets.smiles_frag_ori import drop_names, id2task
import matplotlib.pyplot as plt
import torch
# 设置随机种子
torch.manual_seed(42)

fragmenizer = BRICS_RING_R_Fragmenizer(break_ring=False)

mapping = {'PAD': 0, 'A': 1, 'R': 2, 'N': 3, 'D': 4,
           'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11,
           'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18,
           'Y': 19, 'V': 20}

def parse_args():
    parser = argparse.ArgumentParser(description="GENERATE SAMPLE")
    parser.add_argument("--config", type=str, default='smiles_frag',
                        choices=['smiles_frag_pocket', 'smiles_frag', 'smiles_frag_admet', 'target'],
                        help="Selected a config for this task.")
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--fp16', type=int, default=True)
    parser.add_argument('--adapter', type=bool, default=False)
    parser.add_argument("--lora",
                        action="store_true",
                        help="Run or not.")
    # parser.add_argument('--lora', type=bool, default=False)
    parser.add_argument('--gpu', type=bool, default=True)
    # parser.add_argument('--ckpt', type=str, default='CBLB/no_pretrain')
    parser.add_argument('--ckpt', type=str, default='fraggpt')
    parser.add_argument('--tokenizer', type=str, default='./tokenizer/chembl_pubchem10M/tokenizer.json')
    parser.add_argument('--model_ckpt', type=str, default='model.pt')
    # parser.add_argument('--model_ckpt', type=str, default='CBLB_admet_lora_10.pt')
    parser.add_argument('--n_generated', type=int, default=500, help='the number of sample')
    parser.add_argument('--batch_size', type=int, default=64, help='the number of sample')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--topp', type=float, default=0.96, help='最高积累概率')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--pdb', type=str, default='/home/pengbingxin/pbx/fraggpt/data/CBLB/8gcy_nx1607_sc_pre_protein_fix.pdb')  # './data/menin/pocket.pdb'  BTK.pdb
    parser.add_argument('--sdf', type=str, default='/home/pengbingxin/pbx/fraggpt/data/malt1_ligand.sdf')  # './data/menin/ligand.sdf'   cs_a4j7bm.sdf
    parser.add_argument('--add_pocket', type=bool, default=False, help='是否加入pocket')
    parser.add_argument('--hard_save', type=bool, default=False, help='完全保留已有片段')
    parser.add_argument('--force_generate', type=bool, default=False, help='完全保留已有片段')
    parser.add_argument('--tta', type=bool, default=False, help='是否TTA推理')
    parser.add_argument('--save_png', type=bool, default=False, help='是否保留生成分子的png，大批量不建议')
    parser.add_argument('--generate_mode', type=str, default='denovo',
                        choices=['denovo', 'linker', 'rgroup', 'scaffold', 'sidechain', 'norm',
                                 'norm_linker', 'norm_rgroup', 'norm_scaffold', 'norm_sidechain']) # mix: generate_mode + norm
    parser.add_argument('--reference', type=str,
                        default='CO[C@H](c(c(C)cn1)n2c1cc(Cl)n2)C.[*]c3cc(Cl)c(C(N4CCCC4)=O)nc3') #CO[C@@H](C)c1c(NC(=O)Nc2cnc(C(=O)N3CCCC3)c(Cl)c2)cnc2cc(Cl)nn12


    parser.add_argument('--outputs_dir', type=str, default='outputs/pngs')
    parser.add_argument('--save_dir', type=str, default='./outputs/linker100002024319.csv')
    args = parser.parse_args()
    return args, parser

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

def load_ckpt(ckpt, model):
    assert os.path.exists(ckpt), 'checkpoint no exists! '
    pretrained_dict = torch.load(ckpt, map_location='cpu')
    model_dict = model.state_dict()
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
    model.load_state_dict(model_dict, strict=False)
    print(f"  =====================================================================================")
    print(f"  Load pretrained model from {ckpt}")
    print(f"  =====================================================================================")
    return model

def updata_cfg(cfg, args):
    for k, v in args.__dict__.items():
        cfg[k] = v
    return cfg

def get_frag_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    frags, _ = fragmenizer.fragmenize(mol)
    frags = Chem.GetMolFrags(frags, asMols=True)
    res = [Chem.MolToSmiles(frag) for frag in frags]
    return res

def increment_match(match, k):
    """将匹配的模式加 5"""
    return f"[{int(match.group(1)) + k}*]"

def sub_point_id(string, ts_dict):
    last_left = 0
    i = 0
    while i<len(string):
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

def smiles_augment(frags_list):
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

def prefix2tensor(cfg, n_sample, tokenizer):
    if cfg.generate_mode == 'denovo':
        ori_frags_list = []
    elif cfg.generate_mode == 'linker':
        refs = cfg.reference.split('.')
        assert len(refs) == 2
        frag1 = refs[1]
        frag2 = refs[0]
        frags_list_1 = get_frag_smiles(frag1)
        all_idx_frag1 = find_all_idx(frags_list_1)
        if len(all_idx_frag1) == 0:
            all_idx_frag1 = [0]

        frags_list_2 = get_frag_smiles(frag2)
        all_idx_frag2 = find_all_idx(frags_list_2)

        new_frags_list_2 = []
        pat = r'\[(\d+)\*\]'
        for m, frag in enumerate(frags_list_2):
            new_frags_list_2.append(re.sub(pat, lambda match: increment_match(match, max(all_idx_frag1)), frag))

        all_frag_list = frags_list_1 + new_frags_list_2

        num_breakpoint = 0
        for string in all_frag_list:
            for k, s in enumerate(string):
                if s == '*' and (k == len(string) - 1 or string[k + 1] != ']'):
                    num_breakpoint += 1
        assert num_breakpoint == 2

        select_id = 0

        pat = r'\[(\d+)\*\]'
        new_all_frag_list = []
        for m, frag in enumerate(all_frag_list):
            new_all_frag_list.append(re.sub(pat, lambda match: f"[{int(match.group(1)) + num_breakpoint}*]" if int(match.group(1)) > select_id else match.group(0), frag))

        _ori_frags_list = []
        for j, string in enumerate(new_all_frag_list):
            i = 0
            while i < len(string):
                if string[i] == '*' and (i == len(string) - 1 or string[i + 1] != ']'):
                    select_id += 1
                    string = string[:i] + f'[{select_id}*]' + string[i + 1:]
                    add_len = len(f'[{select_id}*]')
                    i = i - 1 + add_len + 1
                else:
                    i += 1
            _ori_frags_list.append(string)

        if cfg.tta:
            ori_frags_list = smiles_augment(_ori_frags_list)
        else:
            ori_frags_list = _ori_frags_list
        np.random.shuffle(ori_frags_list)

    elif cfg.generate_mode == 'rgroup':
        refs = cfg.reference.split('.')
        assert len(refs) == 1
        frags_list_1 = get_frag_smiles(refs[0])
        all_idx_frag1 = find_all_idx(frags_list_1)
        np.random.shuffle(all_idx_frag1)

        select_id = 0

        pat = r'\[(\d+)\*\]'
        new_frags_list_1 = []
        for m, frag in enumerate(frags_list_1):
            new_frags_list_1.append(re.sub(pat, lambda match: f"[{int(match.group(1))+1}*]" if int(match.group(1)) > select_id else match.group(0), frag))

        _ori_frags_list = []
        for j, string in enumerate(new_frags_list_1):
            i = 0
            while i < len(string):
                if string[i] == '*' and (i == len(string) - 1 or string[i + 1] != ']'):
                    select_id += 1
                    string = string[:i] + f'[{select_id}*]' + string[i + 1:]
                    add_len = len(f'[{select_id}*]')
                    i = i - 1 + add_len + 1
                else:
                    i += 1
            _ori_frags_list.append(string)

        if cfg.tta:
            ori_frags_list = smiles_augment(_ori_frags_list)
        else:
            ori_frags_list = _ori_frags_list
        np.random.shuffle(ori_frags_list)

    elif cfg.generate_mode == 'scaffold':
        refs = cfg.reference.split('.')
        all_frag_list = []
        for ref in refs:
            temp_frag_list = get_frag_smiles(ref)
            if len(all_frag_list) > 0:
                temp_idx_frag = find_all_idx(all_frag_list)
                if len(temp_idx_frag) == 0:
                    temp_idx_frag = [0]
            else:
                temp_idx_frag = [0]
            new_frags_list_2 = []
            pat = r'\[(\d+)\*\]'
            for m, frag in enumerate(temp_frag_list):
                new_frags_list_2.append(re.sub(pat, lambda match: increment_match(match, max(temp_idx_frag)), frag))
            all_frag_list += new_frags_list_2

        all_idx_frag1 = find_all_idx(all_frag_list)

        num_breakpoint = 0
        for string in all_frag_list:
            for k, s in enumerate(string):
                if s == '*' and (k == len(string) - 1 or string[k + 1] != ']'):
                    num_breakpoint += 1

        next_id = 0
        pat = r'\[(\d+)\*\]'
        new_frags_list_1 = []
        for m, frag in enumerate(all_frag_list):
            new_frags_list_1.append(re.sub(pat, lambda match: f"[{int(match.group(1)) + num_breakpoint}*]" if int(match.group(1)) > next_id else match.group(0), frag))

        _ori_frags_list = []
        for j, string in enumerate(new_frags_list_1):
            i = 0
            while i < len(string):
                if string[i] == '*' and (i == len(string) - 1 or string[i + 1] != ']'):
                    next_id += 1
                    string = string[:i] + f'[{next_id}*]' + string[i + 1:]
                    add_len = len(f'[{next_id}*]')
                    i = i - 1 + add_len + 1
                else:
                    i += 1
            _ori_frags_list.append(string)
        if cfg.tta:
            ori_frags_list = smiles_augment(_ori_frags_list)
        else:
            ori_frags_list = _ori_frags_list
        np.random.shuffle(ori_frags_list)

    elif cfg.generate_mode == 'sidechain':
        refs = cfg.reference.split('.')
        assert len(refs) == 1
        frags_list_1 = get_frag_smiles(refs[0])
        all_idx_frag1 = find_all_idx(frags_list_1)

        num_breakpoint = 0
        for string in frags_list_1:
            for k, s in enumerate(string):
                if s == '*' and (k == len(string)-1 or string[k + 1] != ']'):
                    num_breakpoint += 1

        next_id = 0
        pat = r'\[(\d+)\*\]'
        new_frags_list_1 = []
        for m, frag in enumerate(frags_list_1):
            new_frags_list_1.append(re.sub(pat, lambda match: f"[{int(match.group(1)) + num_breakpoint}*]" if int(match.group(1)) > next_id else match.group(0), frag))

        _ori_frags_list = []
        for j, string in enumerate(new_frags_list_1):
            i = 0
            while i < len(string):
                if string[i] == '*' and (i == len(string)-1 or string[i + 1] != ']'):
                    next_id += 1
                    string = string[:i] + f'[{next_id}*]' + string[i + 1:]
                    add_len = len(f'[{next_id}*]')
                    i = i - 1 + add_len + 1
                else:
                    i += 1
            _ori_frags_list.append(string)

        if cfg.tta:
            ori_frags_list = smiles_augment(_ori_frags_list)
        else:
            ori_frags_list = _ori_frags_list
        np.random.shuffle(ori_frags_list)

    elif cfg.generate_mode == 'norm':
        refs = cfg.reference.split('.')
        breakpoint_id = 1
        ori_frags_list = []
        for string in refs:
            i = 0
            while i < len(string):
                if string[i] == '*' and (string[i + 1] == ']' and string[i - 1] == '['):
                    string = string[:i] + str(breakpoint_id) + '*' + string[i+1:]
                    breakpoint_id += 1
                    add_len = len(str(breakpoint_id) + '*')
                    i = i - 1 + add_len + 1
                else:
                    i += 1
            ori_frags_list.append(string)
    else:
        raise ValueError(f"不存在生成模式{cfg.generate_mode}")

    if cfg.hard_save and cfg.generate_mode not in ['denovo']:
        last_frag = ori_frags_list[-1]
        last_all_idx = find_all_idx([last_frag])
        np.random.shuffle(last_all_idx)
        rand_idx = last_all_idx[0]
        while 1:
            mol = Chem.MolFromSmiles(last_frag)
            new_last_frag = Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
            if new_last_frag.endswith(f'[{rand_idx}*]'):
                break
        ori_frags_list[-1] = new_last_frag

    # if len(ori_frags_list) == 1:
    #     input_frags = ori_frags_list[0] + '<sep>'
    # else:
    #     input_frags = '<sep>'.join(ori_frags_list)
    if cfg.generate_mode == 'denovo':
        input_frags = '<sep>'.join(ori_frags_list)
    else:
        input_frags = '<sep>'.join(ori_frags_list) + '<sep>'
    token_batch = tokenizer(input_frags, truncation=True)['input_ids'][:-1]
    input_tensor = torch.LongTensor(token_batch)
    if cfg.gpu:
        input_tensor = input_tensor.unsqueeze(0).repeat(n_sample, 1).cuda()
    else:
        input_tensor = input_tensor.unsqueeze(0).repeat(n_sample, 1)

    return input_tensor, ori_frags_list

def get_refligand_admet(args, smiles):
    with open('admet_predictor/temp/temp_gen.txt', 'w') as f:
        f.write(smiles + '\n')
    with open(os.path.join('save', args.ckpt, 'mean_std.json'), 'r') as f:
        mean_std = json.load(f)
    admet_df = admet_infer2()
    # python_executable = sys.executable
    # os.system(f"{python_executable} admet_predictor/inference_self.py")
    # admet_df = pd.read_csv("/mnt/e/tangui/SMILES_NEW/admet_predictor/temp/temp_g.csv")
    mol = Chem.MolFromSmiles(admet_df.iloc[0, 0])
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    print('参考分子mw: ', mw)
    print('参考分子tpsa: ', tpsa)
    admet = np.array(admet_df.iloc[0, 1:].tolist() + [mw, tpsa], dtype=np.float16)
    admet_prop = []
    for k in range(admet.shape[0]):
        task = id2task[k]
        if task in drop_names:
            continue
        if task in mean_std:
            mean = mean_std[task]['mean']
            std = mean_std[task]['std']
            admet_prop.append(float((admet[k] - mean) / std))
        else:
            admet_prop.append(float(admet[k]))

    return admet_prop


def sample_from_topp_independent(args, logits):
    logits = F.softmax(logits, dim=-1)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)
    sorted_indices_to_remove = cumulative_probs > args.topp

    # make sure at least have one point to sample
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = 0.0
    token_prob = logits / logits.sum(dim=-1, keepdim=True)
    token_prob = token_prob.type(torch.float32)
    token_dist = D.Categorical(token_prob)
    next_token = token_dist.sample((1,)).permute(1, 0)
    return next_token

def generate_next_token(model, input_ids, condition, args, unk_id, end_id,tokenizer):
    """
    对于给定的上文，生成下一个单词
    """
    # outputs = model(input_ids=input_ids)
    with torch.no_grad():
        if args.add_pocket:
            outputs = model(input_ids=input_ids,
                            pocket_seq=condition['pocket_seq'],
                            pocket_edge_type=condition['pocket_edge_type'],
                            pocket_dis=condition['pocket_dis'],
                            admet=condition['admet'] if 'admet' in condition else None)
        else:
            outputs = model(input_ids=input_ids,
                            admet=condition['admet'] if 'admet' in condition else None)

    logits = outputs['logits']
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[:, -1, :]
    next_token_logits = next_token_logits / args.temperature
    next_token_id = sample_from_topp_independent(args, next_token_logits)
    if args.force_generate:
        next_token_id = force_generate(next_token_id, input_ids, end_id, next_token_logits, tokenizer)
    return next_token_id

def check(string):
    all_ids = find_all_idx2([string])
    for k, v in all_ids.items():
        if v < 2:
            return False
    return True

def force_generate(next_token_id, input_ids, end_id, logits, tokenizer):
    mask = (next_token_id == end_id).squeeze()
    cur_seqs = torch.cat((input_ids, next_token_id), dim=1)
    result = tokenizer.batch_decode(cur_seqs, skip_special_tokens=True)
    index = []
    for i, res in enumerate(result):
        if mask[i] == True:
            if not check(res):
                index.append(i)

    for idx in index:
        values, indices = torch.topk(torch.softmax(logits[idx:idx+1], dim=-1), k=2)
        next_token_id[idx] = int(indices[0, -1])
    return next_token_id

def timeout_handler(signum, frame):
    raise TimeoutError('Code execution timed out')

def generate(tokenizer, input_ids, condition, model, args):
    end_id = tokenizer.convert_tokens_to_ids("</s>")
    unk_id = tokenizer.unk_token_id
    cur_len = input_ids.shape[1]
    end_tensor = torch.zeros((input_ids.shape[0]), dtype=torch.bool, device=input_ids.device)
    while True:
        next_token_id = generate_next_token(model=model,
                                            input_ids=input_ids,
                                            condition=condition,
                                            args=args,
                                            unk_id=unk_id,
                                            end_id=end_id,
                                            tokenizer=tokenizer)

        end_tensor |= (next_token_id == end_id).squeeze()
        next_token_id[end_tensor] = end_id
        input_ids = torch.cat((input_ids, next_token_id), dim=1)
        cur_len += 1
        # print(input_ids.shape)
        if end_tensor.sum() == input_ids.shape[0]:
            break
        if input_ids.shape[1] > args.max_length:
            break
    # result = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return input_ids

def get_distance_matrix(pos):
    assert pos.shape[1] == 3, 'The shape of pos is error! '
    return torch.pow((pos.unsqueeze(1) - pos.unsqueeze(0)), 2).sum(-1) ** 0.5

def save_output(args, all_sample, tanimoto_sim=None):
    # graph_save_dir = os.path.join(args.outputs_dir, 'graph_imgs')
    graph_save_dir = args.outputs_dir
    os.system(f'rm -rf {graph_save_dir}')
    # if os.path.exists(graph_save_dir):
    #     shutil.rmtree(graph_save_dir)
    os.makedirs(graph_save_dir, exist_ok=True)
    id2smi = {}
    for i, smi in enumerate(all_sample):
        if tanimoto_sim is None:
            graph_save_path = os.path.join(graph_save_dir, f'smiles_{i}.png')
        else:
            norm_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            tanimoto = tanimoto_sim[norm_smi]
            graph_save_path = os.path.join(graph_save_dir, f'{tanimoto}_smiles_{i}.png')
        # id2smi[f'smiles_{i}'] = smi
        id2smi[smi] = tanimoto
        mol = Chem.MolFromSmiles(smi)
        Draw.MolToFile(
            mol,  # mol对象
            graph_save_path,  # 图片存储地址
            size=(640, 640),
            kekulize=True,
            wedgeBonds=True,
            imageType=None,
            fitImage=False,
            options=None,
        )
    with open('./outputs/generate_smiles.json', 'w') as fw:
        json.dump(id2smi, fw, indent=2)

def main():
    # args & config
    args, parser = parse_args()
    cfg = args_parse(f'{args.config}.yml')
    # model & tokenizer
    cfg = updata_cfg(cfg, args)
    tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
        cfg.tokenizer,
        model_max_length=cfg.DATA.MAX_SMILES_LEN
    )
    if cfg.add_pocket:
        model = FragSmilesPocketGPT(cfg, Tokenizer=tokenizer)
    else:
        model = FragSmilesGPT(cfg, Tokenizer=tokenizer)
    if cfg.adapter:
        adapter_config = PfeifferConfig(
            original_ln_before=True, original_ln_after=True, residual_before_ln=True,
            adapter_residual_before_ln=False, ln_before=False, ln_after=False,
            mh_adapter=False, output_adapter=True, non_linearity=cfg.MODEL.ADAPTER.ACTIVATION,
            reduction_factor=cfg.MODEL.ADAPTER.REDUCTION_FACTOR, cross_adapter=False)
        adapter_name = cfg.MODEL.ADAPTER.adapter_name
        model.add_adapter(adapter_name, config=adapter_config)
        model.train_adapter(adapter_name)
        model.set_active_adapters(adapter_name)

    if cfg.lora:
        print('use LoRA')
        # target_modules = find_all_linear_names(model)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            # target_modules=target_modules,
            inference_mode=True,
            r=cfg.MODEL.PEFT.r,
            lora_alpha=cfg.MODEL.PEFT.lora_alpha,
            bias="none",
            lora_dropout=cfg.MODEL.PEFT.lora_dropout)

        # self.model = inject_adapter_in_model(lora_config, self.model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if cfg.fp16:
        model = model.half()
    if cfg.gpu:
        model = model.cuda()

    checkpoint = f'./save/{args.ckpt}'
    model = load_ckpt(os.path.join(checkpoint, f'{args.model_ckpt}'), model)
    model.eval()

    # condition
    radius = 20.0
    smiles, protein_dict = pdb_to_pocket_data(pdb_file=args.pdb,
                                              bbox_size=radius,
                                              mol_file=args.sdf,
                                              perturb=False)

    print('参考smiles：', smiles)
    if cfg.MODEL.ADMET_ENCODER.use_admet:
        
        # smiles = Chem.MolToSmiles(Chem.SDMolSupplier(args.sdf)[0])
        smiles = 'NS(=O)(C1=CC=C(NC2=NC=NC(C3=CC=C(F)C=C3OC)=N2)C=C1)=O'
        admet = get_refligand_admet(args, smiles)
        admet_prop = torch.tensor(admet)

    num_types = len(mapping)
    res_seq = protein_dict['res_seq']
    res_seq = torch.LongTensor([mapping[i] for i in res_seq])
    res_edge_type = res_seq.view(-1, 1) * num_types + res_seq.view(1, -1)
    res_dis = get_distance_matrix(protein_dict['pkt_node_xyz'])

    gen_mode = cfg.generate_mode.split('_')
    mix = False
    if gen_mode[0] == 'norm' and len(gen_mode) > 1:
        mix = True
    generated_smiles_list = []
    pattern = r"<s>(.*?)</s>"
    cur_count = 0
    t_start = time.time()
    # 超时处理
    signal.signal(signal.SIGALRM, timeout_handler)
    with trange(cfg.n_generated) as pbar:
        while cur_count < cfg.n_generated:
            t1 = time.time()
            timeout_seconds = 30  # 自回归某一步推理时长大于30s，则跳出
            signal.alarm(timeout_seconds)
            try:
                if cfg.n_generated - cur_count < cfg.batch_size:
                    n_sample = cfg.n_generated - cur_count
                else:
                    n_sample = cfg.batch_size
                cur_count += n_sample
                # Generate from "<s>" so that the next token is arbitrary.
                # smiles_start = input_tensor[j * (args.batch_size): (j + 1) * args.batch_size]
                if mix:
                    if cur_count < cfg.n_generated * 0.6:
                        cfg.generate_mode = gen_mode[0]
                        # continue
                    else:
                        cfg.generate_mode = gen_mode[1]
                        # continue
                smiles_start, ori_frags_list = prefix2tensor(cfg, n_sample, tokenizer)
                if cfg.gpu:
                    condition = {'pocket_seq': res_seq.unsqueeze(0).repeat(n_sample, 1).cuda(),
                                 'pocket_edge_type': res_edge_type.unsqueeze(0).repeat(n_sample, 1, 1).cuda(),
                                 'pocket_dis': res_dis.unsqueeze(0).repeat(n_sample, 1, 1).cuda()}
                else:
                    condition = {'pocket_seq': res_seq.unsqueeze(0).repeat(n_sample, 1),
                                 'pocket_edge_type': res_edge_type.unsqueeze(0).repeat(n_sample, 1, 1),
                                 'pocket_dis': res_dis.unsqueeze(0).repeat(n_sample, 1, 1)}
                if cfg.MODEL.ADMET_ENCODER.use_admet:
                    if cfg.gpu:
                        condition['admet'] = admet_prop.unsqueeze(0).repeat(n_sample, 1).cuda()
                    else:
                        condition['admet'] = admet_prop.unsqueeze(0).repeat(n_sample, 1).cuda()
                if cfg.fp16:
                    new_condition = {}
                    for k, v in condition.items():
                        if v.dtype == torch.float32:
                            new_condition[k] = v.half()
                        else:
                            new_condition[k] = v
                    condition = new_condition
                    del new_condition
                generated_ids = generate(tokenizer=tokenizer, input_ids=smiles_start,
                                condition=condition, model=model, args=args)
                gens = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

                gen_smis = []
                for gen in gens:
                    try:
                        frags_list = re.search(pattern, gen).group(1).split('<sep>')
                        gen_smiles = conect_all_fragmens(frags_list, ori_frags_list)
                        # gen_smiles = combine_all_fragmens(frags_list)
                        if gen_smiles == 'fail':
                            continue
                    except:
                        continue
                    gen_smis.append(gen_smiles)
                generated_smiles_list += gen_smis
            except TimeoutError:
                # 超时异常处理逻辑
                print("Code execution timed out")
                continue
            finally:
                signal.alarm(0)
            pbar.update(n_sample)
    # df = pd.DataFrame(generated_smiles_list, columns=['smiles'])
    # df.to_csv(args.save_dir, index=False)

    count = 0
    correct = []
    correct_smiles = []
    for smi in generated_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            count += 1
            correct.append(mol)
            correct_smiles.append(smi)
        else:
            continue
    
    smi = pd.DataFrame(correct_smiles,columns=['smiles'])
    print(smi.head(10))
    

if __name__ == '__main__':
    main()
