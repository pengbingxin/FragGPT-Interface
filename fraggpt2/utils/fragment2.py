import rdkit
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import Recap
import copy
import ipdb


def get_detail_bond(atom_idx_1, atom_idx_2, atominfos, connect_no_ring_dict={}, connect_ring_dict={},
                    two_ring_connect_dict={}):
    """_summary_

    Args:
        atom_idx_1 (int): 原子索引1
        atom_idx_2 (int): 原子索引2
        atominfos (list(list())): 环的信息
        connect_no_ring_dict (dict, optional): _description_. Defaults to {}.
        connect_ring_dict (dict, optional): _description_. Defaults to {}.
        two_ring_connect_dict (dict, optional): _description_. Defaults to {}.
    """
    atom_1_in_ring = False
    atom_2_in_ring = False
    for atom_list in atominfos:
        if (atom_idx_1 in atom_list):
            atom_1_in_ring = True
        if (atom_idx_2 in atom_list):
            atom_2_in_ring = True

    ## 说明两个个原子都不在环上
    if ((atom_1_in_ring is False) and (atom_2_in_ring is False)):
        if atom_idx_1 not in connect_no_ring_dict:
            connect_no_ring_dict[atom_idx_1] = []
        if atom_idx_2 not in connect_no_ring_dict:
            connect_no_ring_dict[atom_idx_2] = []

        connect_no_ring_dict[atom_idx_1].append(atom_idx_2)
        connect_no_ring_dict[atom_idx_2].append(atom_idx_1)

    ## 原子1在环上，原子2在环上, 需要切割
    elif ((atom_1_in_ring is True) and (atom_2_in_ring is True)):
        if atom_idx_1 not in two_ring_connect_dict:
            two_ring_connect_dict[atom_idx_1] = []
        # if atom_idx_2 not in two_ring_connect_dict:
        #     two_ring_connect_dict[atom_idx_2] = []

        two_ring_connect_dict[atom_idx_1].append(atom_idx_2)
        # two_ring_connect_dict[atom_idx_2].append(atom_idx_1)

    ## 原子1在环上，原子2不在环上
    elif ((atom_1_in_ring is True) and (atom_2_in_ring is False)):
        if atom_idx_1 not in connect_no_ring_dict:
            connect_no_ring_dict[atom_idx_1] = []
        if atom_idx_2 not in connect_ring_dict:
            connect_ring_dict[atom_idx_2] = []

        connect_no_ring_dict[atom_idx_1].append(atom_idx_2)
        connect_ring_dict[atom_idx_2].append(atom_idx_1)

    ## 原子1不在环上，原子2在环上
    elif ((atom_2_in_ring is True) and (atom_1_in_ring is False)):
        if atom_idx_2 not in connect_no_ring_dict:
            connect_no_ring_dict[atom_idx_2] = []
        if atom_idx_1 not in connect_ring_dict:
            connect_ring_dict[atom_idx_1] = []

        connect_no_ring_dict[atom_idx_2].append(atom_idx_1)
        connect_ring_dict[atom_idx_1].append(atom_idx_2)


def fragment(smiles=""):
    result = []

    ## 回溯算法
    def transback(bond_list, new_mol, index=0, split_token=0):
        if len(result) < 8:
            if len(bond_list) == index:
                result.append(Chem.MolToSmiles(new_mol))
                return
            else:
                temp_bond_list = bond_list[index]
                if len(temp_bond_list) == 1:
                    temp_split_token = split_token
                    new_mol_copy = copy.deepcopy(new_mol)
                    (atom_idx_1, atom_idx_2) = temp_bond_list[0]
                    bond = new_mol_copy.GetBondBetweenAtoms(atom_idx_1, atom_idx_2)
                    new_mol_copy = Chem.FragmentOnBonds(new_mol_copy, [bond.GetIdx()],
                                                        dummyLabels=[(temp_split_token + 1, temp_split_token + 1)])
                    temp_split_token = temp_split_token + 1

                    transback(bond_list, new_mol_copy, index + 1, temp_split_token)

                else:
                    for i in range(len(temp_bond_list)):
                        temp_split_token = split_token
                        new_mol_copy = copy.deepcopy(new_mol)
                        for j in range(len(temp_bond_list)):
                            if i == j:
                                continue
                            else:
                                (atom_idx_1, atom_idx_2) = temp_bond_list[j]
                                bond = new_mol_copy.GetBondBetweenAtoms(atom_idx_1, atom_idx_2)
                                new_mol_copy = Chem.FragmentOnBonds(new_mol_copy, [bond.GetIdx()], dummyLabels=[
                                    (temp_split_token + 1, temp_split_token + 1)])
                                temp_split_token = temp_split_token + 1

                        transback(bond_list, new_mol_copy, index + 1, temp_split_token)

    mol = None

    ## 先对分子的合法性进行判断
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(e)
    if mol is None:
        return []

    ## 获取环信息
    ring = mol.GetRingInfo()  # rdkit.Chem.rdchem.RingInf
    num_rings = ring.NumRings()  # num_rings (int): 分子中环的数量

    if num_rings > 0:
        atominfos = ring.AtomRings()  # atominfos (tuple(tuple()): 所有 环中原子序号的`元组` 的元组.
        # bondinfos = ring.BondRings() # bondinfos (tuple(tuple()): 所有 环中原子序号的`元组` 的元组.
    else:
        ## 虽然不知道这块有什么用
        atominfos = []
        # bondinfos = []

    res = list(BRICS.FindBRICSBonds(mol))  ## 获取所有的切割方式
    ## res的列表中的第一个成分为键所连的原子序号
    """res = [((6, 5), ('1', '5')),
            ((12, 13), ('4', '5')),
            ((24, 25), ('5', '14')),
            ((5, 4), ('5', '16')),
            ((24, 23), ('5', '16')),
            ((6, 8), ('6', '16')),
            ((12, 11), ('8', '16')),
            ((29, 30), ('14', '16'))]"""

    connect_no_ring_dict = {}  # 原子没有与环相连的结果
    connect_ring_dict = {}  # 原子与环相连的情况
    two_ring_connect_dict = {}  ## 两个环相连的情况

    ## 获取详细的连接信息
    for i in range(len(res)):
        atom_idx_1, atom_idx_2 = res[i][0]
        get_detail_bond(atom_idx_1, atom_idx_2, atominfos, connect_no_ring_dict, connect_ring_dict,
                        two_ring_connect_dict)

    mol_pattern = Chem.MolFromSmarts("NC(=S)")
    matches = mol.GetSubstructMatches(mol_pattern)
    for match in matches:
        get_detail_bond(match[0], match[1], atominfos, connect_no_ring_dict, connect_ring_dict, two_ring_connect_dict)

    mol_pattern = Chem.MolFromSmarts("[C,c]C(=S)[C,c]")
    matches = mol.GetSubstructMatches(mol_pattern)
    for match in matches:
        get_detail_bond(match[0], match[1], atominfos, connect_no_ring_dict, connect_ring_dict, two_ring_connect_dict)
        get_detail_bond(match[1], match[3], atominfos, connect_no_ring_dict, connect_ring_dict, two_ring_connect_dict)

    ## 确保C(=[O,N,S])与芳香环相连时不被去掉
    carbonyl_smart = "[c]C(=[O,N,S])[c]"
    smarts_pattern = Chem.MolFromSmarts(carbonyl_smart)
    carbonyl_matches = mol.GetSubstructMatches(smarts_pattern)

    new_connect_ring_dict = copy.deepcopy(connect_ring_dict)

    for atom_idx_1, connect_list in connect_ring_dict.items():
        ## 如果有多个连接
        if len(connect_list) > 1:
            for atom_idx_2 in connect_list:
                ## 去除冗余的连接
                connect_no_ring_dict[atom_idx_2].remove(atom_idx_1)
                ## 获取键
                bond = mol.GetBondBetweenAtoms(atom_idx_1, connect_list[0])
                ## 如果时单键的话
                if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                    ## 判断环上是不是N，非环原子是不是C；如果是，则保持
                    if ((mol.GetAtomWithIdx(atom_idx_1).GetSymbol() == "C") and (
                            mol.GetAtomWithIdx(atom_idx_2).GetSymbol() == "N")):
                        ## 判断环上是不是N
                        atom_2_in_ring = False
                        for atom_list in atominfos:
                            if (atom_idx_2 in atom_list):
                                atom_2_in_ring = True
                        if atom_2_in_ring:
                            continue

                    ## 判断环上是不是C，非环原子是不是C；
                    elif (mol.GetAtomWithIdx(atom_idx_1).GetSymbol() == "C") and (
                            mol.GetAtomWithIdx(atom_idx_2).GetSymbol() == "C"):
                        Flag = True
                        ## 判断是不是在carbonyl_matches中
                        for carbonyl_match in carbonyl_matches:
                            if (atom_idx_1 in carbonyl_match) and (atom_idx_2 in carbonyl_match):
                                Flag = False
                                break

                        if Flag is True:
                            new_connect_ring_dict[atom_idx_1].remove(atom_idx_2)

        ## 只有一个连接的情况
        elif len(connect_list) == 1:
            atom_idx_2 = connect_list[0]
            bond = mol.GetBondBetweenAtoms(atom_idx_1, connect_list[0])
            ## 忽略环外单键
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                ## TODO
                # if (mol.GetAtomWithIdx(atom_idx_1).GetSymbol()=="C") and (mol.GetAtomWithIdx(atom_idx_2).GetSymbol()=="C"):
                #     pass

                ## 去除冗余的连接
                connect_no_ring_dict[atom_idx_2].remove(atom_idx_1)

                ## 判断环上是不是N，非环原子是不是C；如果是，则保持
                if ((mol.GetAtomWithIdx(atom_idx_1).GetSymbol() == "C") and (
                        mol.GetAtomWithIdx(atom_idx_2).GetSymbol() == "N")):
                    atom_2_in_ring = False
                    for atom_list in atominfos:
                        if (atom_idx_2 in atom_list):
                            atom_2_in_ring = True
                    if atom_2_in_ring:
                        continue

                ## 否则不管是什么情况，都不断键
                else:
                    new_connect_ring_dict[atom_idx_1].remove(atom_idx_2)

    ## 去除空的列表
    new_new_connect_ring_dict = {k: v for k, v in new_connect_ring_dict.items() if len(v) > 0}
    new_connect_no_ring_dict = {k: v for k, v in connect_no_ring_dict.items() if len(v) > 0}
    sorted_connect_no_ring_dict = sorted(new_connect_no_ring_dict.items(), key=len)

    global_bond_list = []
    bond_list = []
    for k, value_list in sorted_connect_no_ring_dict:
        temp_bond_list = []
        for v in value_list:
            if ((k, v) not in global_bond_list) and ((v, k) not in global_bond_list):
                temp_bond_list.append((k, v))
                global_bond_list.append((k, v))

        if len(temp_bond_list) > 0:
            bond_list.append(temp_bond_list)

    for k, value_list in new_new_connect_ring_dict.items():
        temp_bond_list = []
        for v in value_list:
            if ((k, v) not in global_bond_list) and ((v, k) not in global_bond_list):
                temp_bond_list.append((k, v))
                global_bond_list.append((k, v))

        if len(temp_bond_list) > 0:
            bond_list.append(temp_bond_list)

    for k, value_list in two_ring_connect_dict.items():
        temp_bond_list = []
        for v in value_list:
            if ((k, v) not in global_bond_list) and ((v, k) not in global_bond_list):
                temp_bond_list.append((k, v))
                global_bond_list.append((k, v))

        if len(temp_bond_list) > 0:
            bond_list.append(temp_bond_list)

    transback(bond_list, mol)
    result = list(set(result))  ## 去重
    return result


if __name__ == "__main__":
    # smiles = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
    smiles = "NC(=O)[C@H](CCCCNC(=O)CO/N=C/c1ccc(F)cc1)NC(=O)CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCNC(=O)CCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCNC(=O)CCN(C1(C(=O)NO)CCCC1)S(=O)(=O)c1ccc(Oc2ccc(F)cc2)cc1"
    print(fragment(smiles))

    smiles = "FC(F)(F)c1ccc(n2c(C3CCN(CC3)Cc(n4)n(Cc5n(CC)cnc5)c6c4ccc(C(O)=O)c6)c(CCOCCC)nc2)cc1"
    print(fragment(smiles))
