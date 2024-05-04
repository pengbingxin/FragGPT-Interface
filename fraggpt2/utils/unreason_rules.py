import rdkit
from rdkit import Chem
from utils import ring_r_fragmenizer
from rdkit.Chem import Descriptors
import pandas as pd
import sys
import os
import numpy as np
import json
from collections import OrderedDict
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
main_dir = os.path.dirname(os.path.abspath(__file__))

# unreasonable_substructs
unreasonable_substructs_path = './utils/black_substruct/substructs_s.sd'
unreasonable_substructs = Chem.SDMolSupplier(unreasonable_substructs_path, removeHs=False)
unreasonable_substructs = {Chem.MolToSmiles(i): i for i in unreasonable_substructs}

## 脂肪环
aliphatic_cyclic_path = "./utils/black_substruct/aliphatic_cyclic_patten.json"
with open(aliphatic_cyclic_path, 'r') as f:
    json_str = f.read()
aliphatic_cyclic_pattern = json.loads(json_str)  # {smiles: smarts}
## convert smarts to frg_mol
for smiles, smarts in aliphatic_cyclic_pattern.items():
    aliphatic_cyclic_pattern[smiles] = Chem.MolFromSmarts(smarts)

## 一些匹配的模板
pynan_pat = Chem.MolFromSmiles("C1=COC=CC1")
furan_pat = Chem.MolFromSmiles("C1=CC=CO1")
thiophene_pat = Chem.MolFromSmiles("C1=CC=CS1")
pyrrole_pat = Chem.MolFromSmiles("C1=CC=CN1")


def ensure_all_carbon(mol, atominfo):
    """iterating all atom in ring to ensure all atom are `C`

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        atominfo (tuple/list): 一组键序号的`元组`/`列表`.

    Returns:
        (bool): 如果所有的原子都为`C`，那么返回True，反之则返回False.
    """
    for atom_index in atominfo:
        atom = mol.GetAtomWithIdx(atom_index)  # 获取原子序号
        if atom.GetAtomicNum() != 6:
            return False
    return True


def ring_is_aromatic(mol, bondinfo):
    """判断环是否为芳香环（`AROMATIC`），如果是则返回True，反之则返回False.

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        bondinfo (tuple/list): 一组键序号的`元组`.

    Returns:
        is_aromatic (bool): 如果是芳香环（`AROMATIC`）则返回True，反之则返回False.
    """
    is_aromatic = True
    for bond_id in bondinfo:
        bond = mol.GetBondWithIdx(bond_id)  # get bond in molecule with bond indx
        # Because int(rdkit.Chem.rdchem.BondType.AROMATIC) == 12,
        # if int(bond_type) is not equal to 12, that means this bond is not `AROMATIC`
        bond_type = bond.GetBondType()
        if bond_type != Chem.rdchem.BondType.AROMATIC:
            is_aromatic = False
            break
    return is_aromatic


def ring_unsaturated_bond_nums(mol, bondinfo):
    """计算环中`不饱和键`的数量

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        bondinfo (tuple): 一组键序号的`元组`.

    Returns:
        unsaturated_bond_nums (int): 环中`不饱和键`的数量.
    """
    unsaturated_bond_nums = 0
    for bond_id in bondinfo:
        bond = mol.GetBondWithIdx(bond_id)  # bond (rdkit.Chem.rdchem.Bond)
        bond_type = bond.GetBondType()  # rdkit.Chem.rdchem.BondType
        if bond_type != Chem.BondType.SINGLE:
            unsaturated_bond_nums = unsaturated_bond_nums + 1
    return unsaturated_bond_nums


def ring_double_bond_nums(mol, bondinfo):
    """返回环中双键的数量

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        bondinfo (tuple/list): 一组键序号的`元组`或`列表`.

    Returns:
        double_bond_nums (int): 指定原子中双键的数目
    """
    double_bond_nums = 0
    for bond_id in bondinfo:
        bond = mol.GetBondWithIdx(bond_id)
        bond_type = bond.GetBondType()
        if int(bond_type) == 2:
            double_bond_nums = double_bond_nums + 1
    return double_bond_nums


def link_with_hetero_atom(mol, atom_idx_list=[]):
    """判断原子是否与杂原子相连

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子
        atom_idx_list (tuple/list, optional): 一组原子序号的`列表`或`元组`. Defaults to [].

    Returns:
        is_link_with_hetero_atom (float): 如果存在于atom_idx_list中的原子有与杂原子相连，则返回True，反之则返回False
    """
    is_link_with_hetero_atom = False
    for atom_idx in atom_idx_list:
        for neighbor_atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_atom_idx = neighbor_atom.GetIdx()  # 获取邻居原子的序号
            ## 保证邻居原子的序号不在atom_idx_list中(不加这句代码也可以)
            if neighbor_atom_idx in atom_idx_list:
                continue
            neighbor_atom_symbol = neighbor_atom.GetSymbol().upper()
            if neighbor_atom_symbol == 'C':
                continue
            else:
                is_link_with_hetero_atom = True
                return is_link_with_hetero_atom
    return is_link_with_hetero_atom


def aliphatic_cyclic_double_bonds_nums(mol, atominfo, bondinfo):
    """ 计算脂肪环中不饱和键的数量

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子
        atominfo (tuple/list): 一组原子序号的`元组`.
        bondinfo (tuple/list): 一组键序号的`元组`.

    Returns:
        result_dict: 以`字典`的形式返回不饱和键的数量.
    """

    result_dict = {}

    if len(atominfo) == 5:
        ## 确保所有的原子都是碳原子
        Is_All_Carbon = ensure_all_carbon(mol, atominfo)
        if Is_All_Carbon is True:
            Is_Link_With_Hetero = link_with_hetero_atom(mol, atominfo)
            if Is_Link_With_Hetero is False:
                result_dict["五元脂肪碳环含有不合理结构"] = 1

    elif len(atominfo) > 5:
        ## 确保所有原子必须是碳原子
        Is_All_Carbon = ensure_all_carbon(mol, atominfo)
        if Is_All_Carbon is True:
            double_bond_nums = ring_double_bond_nums(mol, bondinfo)
            if double_bond_nums == 2:
                if len(atominfo) == 6:
                    ## [#6H0]表示C上没有H
                    sub_mol = Chem.MolFromSmarts("[#6]1=[#6]-[#6H0]-[#6]=[#6]-[#6H0]-1")
                    matches = mol.GetSubstructMatches(sub_mol)
                    if len(matches) > 0:
                        # 白名单
                        # print("六元环匹配到`[#6]1=[#6]-[#6H0]-[#6]=[#6]-[#6H0]-1`")
                        pass
                    else:
                        result_dict["六元及以上脂肪碳环含有两个双键"] = 1
                else:
                    result_dict["六元及以上脂肪碳环含有两个双键"] = 1
    return result_dict


def fragment_on_bond_ids(mol, bond_ids):
    if len(bond_ids) > 0:
        bond_ids = list(set(bond_ids))
        dummyStart = 1
        dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
        break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)  #
        res = Chem.GetMolFrags(break_mol, asMols=True, sanitizeFrags=False)
    else:
        res = (mol,)
    return res


def atom_all_bonds(mol, atom_idx):
    bond_ids = []
    bonds = mol.GetBonds()
    for bond in bonds:
        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        if atom_idx in [begin_atom_idx, end_atom_idx]:
            bond_ids.append(bond.GetIdx())
    return list(set(bond_ids))


def get_charge(mol):
    """遍历所有原子并计算带电情况, 并返回列表。无法对游离的盐有比较好的判断

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Returns:
        charge_list (list): 所有原子的形式电荷的列表
    """

    charge_list = []
    # 遍历所有原子并计算带电情况
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()  # 获取原子索引
        atom_charge = atom.GetFormalCharge()  # 获取原子带电情况
        charge_list.append(atom_charge)
    return charge_list


def compute_charge(mol, low=0, high=np.inf):
    """遍历所有原子并计算带电情况, 并返回电荷数>=low和电荷数<=high的原子个数。无法对游离的盐有比较好的判断

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        low (int, optional): 最低值. Defaults to 0.
        high (float, optional): 最高值. Defaults to np.inf.

    Returns:
        count (int): 分子中的电荷数>=low和电荷数<=high的原子个数
    """
    if low > high:
        raise ValueError("the value of `high` should greater than the value of `low`")
    charge_list = get_charge(mol)
    count = 0
    for charge in charge_list:
        if charge >= low and charge <= high:
            count = count + 1
    return count


def unreasonable_aliphatic_cyclic(mol, atom_list=[]):
    """计算脂肪环中是否含有不合理结构

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        atom_list (list, optional): the index of atom within box. Defaults to [].

    Returns:
        result_dict(dict): 脂肪环中是否含有不合理结构
    """
    result_dict = {}
    # Has_Carbonyl 初始化为None。
    Has_Carbonyl = None
    for aliphatic_cyclic in aliphatic_cyclic_pattern.values():
        matches = mol.GetSubstructMatches(aliphatic_cyclic)
        if len(matches) > 0:
            for match in matches:

                ## 确保匹配的原子在五元环中
                if len(set(atom_list) & set(match)) != len(match):
                    continue
                Has_Carbonyl = False
                for atom_idx in match:
                    ## 只对中间的"C"进行测试
                    atom = mol.GetAtomWithIdx(atom_idx)
                    if atom.GetSymbol() != "C":
                        continue
                    ## 遍历和中间原子相连的原子
                    for neighbor_atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
                        ## 获取相连原子的序号
                        neighbor_atom_idx = neighbor_atom.GetIdx()
                        ## 如果相连原子不在match中
                        if neighbor_atom_idx not in match:
                            ## 继续判断相连原子是否为S或O原子
                            neighbor_atom_symbol = neighbor_atom.GetSymbol()
                            if (neighbor_atom_symbol == 'O') or (neighbor_atom_symbol == 'S'):
                                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_atom_idx)
                                bond_type = bond.GetBondType()
                                ## 如果键的类型不为双键
                                if bond_type == Chem.rdchem.BondType.DOUBLE:
                                    Has_Carbonyl = True

    ## None is not False
    ## 只记录不合理结构
    if Has_Carbonyl is False:
        result_dict["脂肪环中含有不合理结构"] = 1

    return result_dict


def get_substructure(mol, atom_list=[]):
    """Returns the largest substructure fragment

    Args:
        mol (rdkit.Chem.rdchem.Mol): molecule
        atom_list (list, optional): the index of atom within box. Defaults to [].

    Returns:
        mol1 (rdkit.Chem.rdchem.Mol): molecule
    """

    mol1 = Chem.RWMol(mol)
    total_atom = len(mol1.GetAtoms())
    for atom_index in range(total_atom - 1, -1, -1):
        # exclude the atom that not in atom_list
        if (atom_index not in atom_list):
            mol1.RemoveAtom(atom_index)
            mol1 = Chem.RWMol(mol1)
    return mol1


def fragment_on_unsaturated_bonds(mol):
    """_summary_

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Returns:
        _type_: _description_
    """
    unsaturated_bond_ids = []
    bonds = mol.GetBonds()
    for bond in bonds:
        if int(bond.GetBondType()) != 1:
            unsaturated_bond_ids.append(bond.GetIdx())
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()
            unsaturated_bond_ids.extend(atom_all_bonds(mol, begin_atom_idx))
            unsaturated_bond_ids.extend(atom_all_bonds(mol, end_atom_idx))

    res = fragment_on_bond_ids(mol, unsaturated_bond_ids)
    return res


def fragment_on_ring_bonds(mol):
    fragmenizer = ring_r_fragmenizer.RING_R_Fragmenizer()
    bonds = fragmenizer.get_bonds(mol)
    bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
    res = fragment_on_bond_ids(mol, bond_ids)
    return res


def count_dummies(mol):
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            count += 1
    return count


def get_num_spirocyclic(mol, num_rings_threshold=3):
    """_summary_

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        num_rings_threshold (int, optional): threshold of num_spirocyclic. Defaults to 3.

    Returns:
        num_spirocyclic (int): the num_spirocyclic
    """
    num_spirocyclic = 0
    fragmenizer = ring_r_fragmenizer.RING_R_Fragmenizer()
    dummyStart = 1
    bonds = fragmenizer.get_bonds(mol)
    bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
    bond_ids = list(set(bond_ids))
    dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
    break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
    res = Chem.GetMolFrags(break_mol, asMols=True, sanitizeFrags=False)

    if len(res) < 0:
        res = (mol,)

    num_rings_threshold = 3
    for r in res:
        sssr = Chem.GetSymmSSSR(r)
        if len(sssr) >= num_rings_threshold:
            num_spirocyclic = num_spirocyclic + 1
    return num_spirocyclic


def unreasonable_general(mol):
    """以字典形式返回分子中不合理结构的记录

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Returns:
        result_dict (dict): 不合理结构记录的存储字典.
    """

    ## 默认情况
    result_dict = {
        "质子化原子的个数>1": 0,
        "烯醇式子结构": 0,
        "长链结构（大于等于5个碳原子和含有杂原子的柔性长链）": 0,
        "多并环（>= 3）": 0,
        "九元以上环": 0,
        "C12=CC=CC1=NC=CS2": 0}

    ## 计算质子化的原子的个数
    try:
        charge_count = compute_charge(mol, low=1)
        if charge_count > 1:
            result_dict["质子化原子的个数>1"] = 1
    except Exception as e:
        print(e)
        print("计算`质子化原子的个数`失败")

    # 烯醇式结构
    try:
        ## 注意，所有匹配模式，不要用smiles模式
        ethenol_pat = Chem.MolFromSmarts("[#6]=[#6]-[#8H]")
        ethenol_matches = mol.GetSubstructMatches(ethenol_pat)
        if len(ethenol_matches) > 0:
            result_dict["烯醇式子结构"] = 1
    except Exception as e:
        print("计算`烯醇式子结构`失败")

    # # 长链结构（大于等于5个碳原子和含有杂原子的柔性长链）
    # from rdkit.Chem.Scaffolds import MurckoScaffold
    try:
        res_frag_ring = fragment_on_ring_bonds(mol)
        if len(res_frag_ring) > 0:
            for frag_ring in res_frag_ring:
                sssr = Chem.GetSymmSSSR(frag_ring)
                if len(sssr) == 0:
                    pat = Chem.MolFromSmarts("[C,N;d2][C,N;d2][C,N;d2][C,N;d2][C,N;d2]")
                    long_chain_matches = frag_ring.GetSubstructMatches(pat)
                    if len(long_chain_matches) > 0:
                        result_dict["长链结构（大于等于5个碳原子和含有杂原子的柔性长链）"] = 1
    except Exception as e:
        print("计算`长链结构（大于等于5个碳原子和含有杂原子的柔性长链）`失败")

    # 多并环（>= 3）
    try:
        num_spirocyclic = get_num_spirocyclic(mol, num_rings_threshold=3)
        if num_spirocyclic > 0:
            result_dict["多并环（>= 3）"] = 1
    except Exception as e:
        ## 这个失败是正常的，具体的代码我还没仔细看，主要原因，可能不能按照环来割
        print("计算`多并环（>= 3）`失败")

    # 九元以上环
    try:
        ri = mol.GetRingInfo()
        nMacrocycles = 0
        for x in ri.AtomRings():
            if 8 < len(x):
                nMacrocycles += 1
        if nMacrocycles > 0:
            result_dict["九元以上环"] = 1
    except Exception as e:
        print("计算`九元以上环`失败")

    ## 5并6元环不合理结构 "C12=CC=CC1=NC=CS2"
    try:
        sub_mol = Chem.MolFromSmiles("C12=CC=CC1=NC=CS2")
        sub_mol_matches = mol.GetSubstructMatches(sub_mol)
        if len(sub_mol_matches) > 0:
            result_dict["C12=CC=CC1=NC=CS2"] = 1
    except Exception as e:
        print("计算`C12=CC=CC1=NC=CS2`失败")

    return result_dict


def get_default_result():
    result_dict = {'[H]C([H])([H])/C=C/C': 0,
                   'C=CC=C': 0,
                   'CN1C=CC=CC1': 0,
                   'C=CC=N': 0,
                   'C=C1CCCO1': 0,
                   'c1c[nH]c2ccnc-2n1': 0,
                   'C1=COCCC1': 0,
                   'COC(O)OC': 0,
                   'C=C1C=CNC1': 0,
                   'c1ccc2c(c1)Oc1ccc3c(c1O2)OCO3': 0,
                   'N=CC=N': 0,
                   'C=c1cc2c(cn1)=NN=C2': 0,
                   'C=COC=C': 0,
                   '[H]C([H])([H])C([H])([H])C([H])([H])C([H])([H])C': 0,
                   'C1=NNCNN1': 0,
                   '质子化原子的个数>1': 0,
                   '烯醇式子结构': 0,
                   '长链结构（大于等于5个碳原子和含有杂原子的柔性长链）': 0,
                   '多并环（>= 3）': 0,
                   '九元以上环': 0,
                   'C12=CC=CC1=NC=CS2': 0,
                   '脂肪环中含有不合理结构': 0,
                   '五元脂肪碳环含有不合理结构': 0,
                   '六元及以上脂肪碳环含有两个双键': 0,
                   '大于等于6元环，3个N原子，连接在一起': 0,
                   '四元并环（非氧杂环）': 0,
                   '6元环，非吡喃，非芳香环，含两个及两个以上不饱和键。排除环烯': 0,
                   '5元杂环，非呋喃，噻吩，吡咯，含有两个双键': 0
                   }
    return copy.deepcopy(result_dict)


def get_bond_from_atom_idx(mol, atom_idx_list=[]):
    """根据一组原子序号列表获得其中的键

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        atom_idx_list (list/tuple, optional): 原子索引列表. Defaults to [].

    return:
        bond_dict(dict): [(atom_idx, neighbor_atom_idx):bond_type], 返回键的原子序号id的`元组`的`列表`
    """
    bond_dict = {}
    for atom_idx in atom_idx_list:
        for neighbor_atom in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_atom_idx = neighbor_atom.GetIdx()
            ## 如果邻居原子不在atom_idx_list，则跳过
            if neighbor_atom_idx not in atom_idx_list:
                continue
            temp = (neighbor_atom_idx, atom_idx)
            if atom_idx < neighbor_atom_idx:
                temp = (atom_idx, neighbor_atom_idx)

            ## 如果键已经存在double_bond_list中，就不需要再判断是否为双键
            if temp in bond_dict:
                continue
            else:
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_atom_idx)
                bond_type = bond.GetBondType()
                bond_dict[temp] = bond_type

    return bond_dict


## 这一块可以写的更加通用一些
def get_double_bond_from_atom_idx(mol, atom_idx_list=[]):
    """根据一组原子序号列表获得其中的双键

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        atom_idx_list (list/tuple, optional): 原子索引列表. Defaults to [].

    return:
        double_bond_list(list(tuple)): [(atom_idx, neighbor_atom_idx),], 返回双键的原子序号id的`元组`的`列表`

    """
    double_bond_dict = get_bond_from_atom_idx(mol, atom_idx_list)
    new_double_bond_dict = {}
    for bond, bond_type in double_bond_dict.items():
        if bond_type == Chem.rdchem.BondType.DOUBLE:
            new_double_bond_dict[bond] = bond_type

    return new_double_bond_dict


## 尚未判断
def bond_in_ring(mol, bond_list=[]):
    """判断键是否在环上，如果是则返回True，反之则返回False

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.
        bond_list (list/tuple, optional): 键两端的原子序号索引列表.比如[(1,2),(2,4)]. Defaults to [].

    Returns:
        is_bond_in_ring (bool): 只要有一组键两端的原子序号在环环上，则返回True，反之则返回False.
    """

    is_bond_in_ring = False  # 维护一个变量

    ri = mol.GetRingInfo()  # rdkit.Chem.rdchem.RingInf
    num_rings = ri.NumRings()  # num_rings (int): 分子中环的数量
    atominfos = ri.AtomRings()  # tuple(tuple())

    for id_ring in range(num_rings):
        atominfo_set = set(atominfos[id_ring])  # 环中原子序号的`元组`.
        for bond in bond_list:
            bond_set = set(bond)  # 键两端原子序号的`元组`.
            ## 如果bond_set在atominfo_set中，则键在环上
            if len(atominfo_set & bond_set) == len(bond_set):
                return True
    return is_bond_in_ring


twelve_membered_ring_list = ["C12=COC=C1CC3=COC=C3C2"]
twelve_memberd_white_list = [Chem.MolFromSmiles(smiles) for smiles in twelve_membered_ring_list]


def add_twelve_memberd_fused_ring_white_list(mol):
    """返回与白名单上的十二元环匹配的原子序号列表

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Returns:
        twelve_memberd_match_list (list): 与白名单上的十二元环匹配的原子序号列表
    """
    twelve_memberd_match_list = []
    for sub_mol in twelve_memberd_white_list:
        matches = mol.GetSubstructMatches(sub_mol)
        for match in matches:
            twelve_memberd_match_list.append(match)
    return twelve_memberd_match_list


nine_memberd_path = "./utils/black_substruct/nine_memberd_fused_ring_pattern.json"
with open(nine_memberd_path, "r") as f:
    json_str = f.read()
nine_membered_dict = json.loads(json_str)
nine_membered_replace_dict = {}
for k, v in nine_membered_dict.items():
    new_key = Chem.MolFromSmiles(k)
    new_value = Chem.MolFromSmiles(v)
    nine_membered_replace_dict[new_key] = new_value


def add_nine_memberd_fused_ring_white_list(mol):
    """返回与白名单上的十二元环匹配的原子序号列表

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Returns:
        nine_memberd_match_list (list): 与白名单上的九元环匹配的原子序号列表
    """
    nine_memberd_match_list = []
    for sub_mol, new_sub_mol in nine_membered_replace_dict.items():
        matches = mol.GetSubstructMatches(sub_mol)
        for match in matches:
            nine_memberd_match_list.append(match)

    return nine_memberd_match_list


def unreasonable_all_rules(mol):
    """以字典形式返回分子中不合理结构的记录

    Args:
        mol (rdkit.Chem.rdchem.Mol): 分子.

    Returns:
        result_dict (dict): 不合理结构记录的存储字典.
    """
    result_dict = get_default_result()  # 维护一个字典来记录不合理的子结构

    # 黑名单结构过滤
    mol_addHs = Chem.AddHs(mol)
    for sub_smiles, sub_mol in unreasonable_substructs.items():
        ## "[H]C([H])=CC","C=C1CCCC1","[H]C(=O)NC"已经不在是过滤条件了
        if sub_smiles in ["[H]C([H])=CC", "C=C1CCCC1", "[H]C(=O)NC"]:
            continue

        # match = mol.GetSubstructMatch(sub_mol) #只会返回一个结果
        matches = mol_addHs.GetSubstructMatches(sub_mol)  # 返回一组匹配结果

        if len(matches) == 0:
            result_dict[sub_smiles] = 0

        else:
            Is_At_Ring = False
            ## 如果`sub_smiles`在白名单中
            if sub_smiles in ["[H]C([H])([H])/C=C/C", "C=CC=N", "C=CC=C", "C=COC=C", "N=CC=N"]:
                ## 遍历所有的匹配对象，如果存在匹配原子的其中一个双键不在环上，则令`Is_At_Ring = False`, 并跳出循环
                for match in matches:
                    double_bond_dict = get_double_bond_from_atom_idx(mol_addHs, match)
                    double_bond_list = list(double_bond_dict.keys())
                    if bond_in_ring(mol_addHs, double_bond_list) is True:
                        Is_At_Ring = True
                    else:
                        Is_At_Ring = False
                        break

                ## 如果`Is_At_Ring = False`,那么说明双键不在环上，那么`result_dict[sub_smarts] = 1`
                if Is_At_Ring is not False:
                    result_dict[sub_smiles] = 0
                else:
                    result_dict[sub_smiles] = 1

            else:
                if len(matches) > 0:
                    result_dict[sub_smiles] = 1
                else:
                    result_dict[sub_smiles] = 0

    # 不合理规则过滤
    general_dict = unreasonable_general(mol)  # general_dict (dict): 通用不合理结构的字典记录
    result_dict.update(general_dict)  # 更新字典

    ri = mol.GetRingInfo()  # rdkit.Chem.rdchem.RingInf
    num_rings = ri.NumRings()  # num_rings (int): 分子中环的数量

    ## 对9元并环和12元并环进行操作
    nine_memberd_match_list = add_nine_memberd_fused_ring_white_list(mol)
    twelve_memberd_match_list = add_twelve_memberd_fused_ring_white_list(mol)
    if num_rings >= 1:
        ## 匹配pynan
        pynan_matches = mol.GetSubstructMatches(pynan_pat)  # 元组. 如果存在，元组的长度大于0.

        ## 匹配furan
        furan_matches = mol.GetSubstructMatches(furan_pat)  # 元组. 如果存在，元组的长度大于0.

        ## 匹配thiophene
        thiophene_matches = mol.GetSubstructMatches(thiophene_pat)  # 元组. 如果存在，元组的长度大于0.

        ## 匹配pyrrole
        pyrrole_matches = mol.GetSubstructMatches(pyrrole_pat)  # 元组. 如果存在，元组的长度大于0.

        atominfos = ri.AtomRings()  # atominfos (tuple(tuple()): 所有 环中原子序号的`元组` 的元组.
        bondinfos = ri.BondRings()  # bondinfos (tuple(tuple()): 所有 环中原子序号的`元组` 的元组.

        ## 注意！！！遍历环，只需要将该过滤的子结构和黑名单中的元素加入即可，不要把白名单和不该过滤的子结构加入
        for id_ring in range(num_rings):
            atominfo = atominfos[id_ring]  # 环中原子序号的`元组`.
            bondinfo = bondinfos[id_ring]  # 环中键序号的`元组`.
            # 判断是否为脂肪环
            is_aromatic = ring_is_aromatic(mol, bondinfo)
            # 脂肪环
            if is_aromatic is False:
                # 脂肪环中含有不合理结构
                unreasonable_aliphatic_dict = unreasonable_aliphatic_cyclic(mol, atominfo)
                result_dict.update(unreasonable_aliphatic_dict)

                # 五元脂肪碳环含有不合理结构；六元及以上脂肪碳环含有两个双键
                aliphatic_cyclic_result = aliphatic_cyclic_double_bonds_nums(mol, atominfo, bondinfo)
                result_dict.update(aliphatic_cyclic_result)

                # 四元并环（非氧杂环）
                if len(atominfo) == 4 and ensure_all_carbon(mol, atominfo):
                    atominfo_set1 = set(atominfo)  # 环中原子序号的`集合`，（用集合进行去重）
                    bondinfo_set1 = set(bondinfo)  # 环中键序号的`集合`，（用集合进行去重）

                    ## 再次遍历环，获得邻接环
                    for adj_id_ring in range(num_rings):

                        ## 如果是同一个环就跳过，否则则判断邻接环与目标环之间是否有两个原子以及一条边相连
                        if adj_id_ring == id_ring:
                            continue
                        else:
                            atominfo_set2 = set(atominfos[adj_id_ring])
                            bondinfo_set2 = set(bondinfos[adj_id_ring])

                            # 如果邻接环与目标环之间是否有两个原子以及一条边相连，则为四元并环
                            if len(atominfo_set1 & atominfo_set2) == 2 and len(bondinfo_set1 & bondinfo_set2) == 1:
                                Is_Four_Membered_Fused_Ring = True
                                ## 此外必须确保邻接环是非氧杂环
                                for atom_idx in atominfo_set2:
                                    atom = mol.GetAtomWithIdx(atom_idx)
                                    if atom.GetAtomicNum() == 8:
                                        Is_Four_Membered_Fused_Ring = False
                                        break
                                if Is_Four_Membered_Fused_Ring == 1:
                                    result_dict["四元并环（非氧杂环）"] = 1

            # 6元环，非吡喃， 非芳香环,含两个及两个以上不饱和键。排除环烯
            if len(atominfo) == 6:
                ring_is_pynan = False
                for match in pynan_matches:
                    if set(atominfo) == set(match):
                        ring_is_pynan = True
                        break

                if not ring_is_pynan:
                    ## 判断是否为芳香环，如果不是，则继续判断是否含有两个不饱和键
                    is_aromatic = ring_is_aromatic(mol, bondinfo)
                    if not is_aromatic:
                        unsaturated_bond_nums = ring_unsaturated_bond_nums(mol, bondinfo)
                        if unsaturated_bond_nums >= 2:
                            ## 匹配上白名单结构（1,4-环二烯，要求3,6位是杂原子）
                            sub_mol_1 = Chem.MolFromSmarts("[#6]1=[#6]-[!C]-[#6]=[#6]-[!C]-1")
                            matches_1 = mol.GetSubstructMatches(sub_mol_1)
                            ## 匹配上白名单结构（1,4-环二烯，要求3位是杂原子）
                            sub_mol_2 = Chem.MolFromSmarts("[#6]1=[#6]-[!C]-[#6]=[#6]-[#6]-1")
                            matches_2 = mol.GetSubstructMatches(sub_mol_2)
                            if (len(matches_1) > 0) or (len(matches_2) > 0):
                                ## 只对不合理的结果进行记录
                                # print("环在白名单上")
                                pass
                            else:
                                # 判断是原子序号是否在环的白名单列表中
                                Is_In_White_List = False
                                for nine_memberd_ring in nine_memberd_match_list:
                                    if len(set(nine_memberd_ring) & set(atominfo)) == len(atominfo):
                                        Is_In_White_List = True
                                        break
                                for twelve_memberd_ring in twelve_memberd_match_list:
                                    if len(set(twelve_memberd_ring) & set(atominfo)) == len(atominfo):
                                        Is_In_White_List = True
                                        break

                                # 如果`Is_In_White_List = False`，只对不合理的结构进行记录
                                if Is_In_White_List is False:
                                    result_dict["6元环，非吡喃，非芳香环，含两个及两个以上不饱和键。排除环烯"] = 1
                                else:
                                    ## 只对不合理的结果进行记录
                                    # print("环在白名单上")
                                    pass

            # 大于等于6元环, 3个N原子(不管用什么键连在一起),连接在一起
            if len(atominfo) >= 6:
                three_nitrogen_smarts = "[#7]~[#7]~[#7]"  # ~表示任意键(包括单键，双键，三键)
                three_nitrogen_mol = Chem.MolFromSmiles(three_nitrogen_smarts)
                three_nitrogen_math = mol.GetSubstructMatches(three_nitrogen_mol)  #
                ## 只对不合理的结果进行记录
                if len(three_nitrogen_math) > 0:
                    result_dict["大于等于6元环，3个N原子，连接在一起"] = 1

            ## 这段逻辑也有点啰嗦, 暂时先这样吧，不大想改
            # 5元杂环，非呋喃，噻吩，吡咯，含有两个双键
            # 先判断是否为五元杂环
            Flag = ensure_all_carbon(mol, atominfo)
            if Flag is False:
                if len(atominfo) == 5:
                    ring_is_furan = False
                    for match in furan_matches:
                        if set(atominfo) == set(match):
                            ring_is_furan = True
                            break

                    if not ring_is_furan:
                        ring_is_thiophene = False
                        for match in thiophene_matches:
                            if set(atominfo) == set(match):
                                ring_is_thiophene = True
                                break

                        if not ring_is_thiophene:
                            ring_is_pyrrole = False
                            for match in pyrrole_matches:
                                if set(atominfo) == set(match):
                                    ring_is_pyrrole = True
                                    break

                            if not ring_is_pyrrole:
                                double_bond_nums = ring_double_bond_nums(mol, bondinfo)
                                if double_bond_nums == 2:
                                    result_dict["5元杂环，非呋喃，噻吩，吡咯，含有两个双键"] = 1

    ## double check
    if "脂肪环中含有不合理结构" not in result_dict.keys():
        result_dict["脂肪环中含有不合理结构"] = 0
    if "五元脂肪碳环含有不合理结构" not in result_dict.keys():
        result_dict["五元脂肪碳环含有不合理结构"] = 0
    if "六元及以上脂肪碳环含有两个双键" not in result_dict.keys():
        result_dict["六元及以上脂肪碳环含有两个双键"] = 0
    if "四元并环（非氧杂环）" not in result_dict.keys():
        result_dict["四元并环（非氧杂环）"] = 0
    if "6元环，非吡喃，非芳香环，含两个及两个以上不饱和键。排除环烯" not in result_dict.keys():
        result_dict["6元环，非吡喃，非芳香环，含两个及两个以上不饱和键。排除环烯"] = 0
    if "大于等于6元环，3个N原子，连接在一起" not in result_dict.keys():
        result_dict["大于等于6元环，3个N原子，连接在一起"] = 0
    if "5元杂环，非呋喃，噻吩，吡咯，含有两个双键" not in result_dict.keys():
        result_dict["5元杂环，非呋喃，噻吩，吡咯，含有两个双键"] = 0

    return result_dict


if __name__ == "__main__":
    smi = "COc1cc(O)cc(C(=O)O)c1CC(=O)Nc1ccccc1"
    mol = Chem.MolFromSmiles(smi)
    unreasonable_contained = unreasonable_all_rules(mol)
    print(unreasonable_contained)