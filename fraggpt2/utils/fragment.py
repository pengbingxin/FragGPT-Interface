from rdkit.Chem import BRICS
import numpy as np
import re
import copy
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit import Chem
from rdkit.Chem.rdchem import BondType

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

class BRICS_Fragmenizer():
    def __inti__(self):
        self.type = 'BRICS_Fragmenizers'

    def get_bonds(self, mol):
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]
        return bonds

    def fragmenize(self, mol, dummyStart=1):
        # get bonds need to be break
        bonds = [bond[0] for bond in list(FindBRICSBonds(mol))]

        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd


def get_rings(mol):
    rings = []
    for ring in list(Chem.GetSymmSSSR(mol)):
        ring = list(ring)
        rings.append(ring)
    return rings


def get_other_atom_idx(mol, atom_idx_list):
    ret_atom_idx = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atom_idx_list:
            ret_atom_idx.append(atom.GetIdx())
    return ret_atom_idx


def find_parts_bonds(mol, parts):
    ret_bonds = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            i_part = parts[i]
            j_part = parts[j]
            for i_atom_idx in i_part:
                for j_atom_idx in j_part:
                    bond = mol.GetBondBetweenAtoms(i_atom_idx, j_atom_idx)
                    if bond is None:
                        continue
                    ret_bonds.append((i_atom_idx, j_atom_idx))
    return ret_bonds


class RING_R_Fragmenizer():
    def __init__(self):
        self.type = 'RING_R_Fragmenizer'

    def bonds_filter(self, mol, bonds):
        filted_bonds = []
        for bond in bonds:
            bond_type = mol.GetBondBetweenAtoms(bond[0], bond[1]).GetBondType()
            if not bond_type is BondType.SINGLE:
                continue
            f_atom = mol.GetAtomWithIdx(bond[0])
            s_atom = mol.GetAtomWithIdx(bond[1])
            if f_atom.GetSymbol() == '*' or s_atom.GetSymbol() == '*':
                continue
            if mol.GetBondBetweenAtoms(bond[0], bond[1]).IsInRing():
                continue
            filted_bonds.append(bond)
        return filted_bonds

    def get_bonds(self, mol):
        bonds = []
        rings = get_rings(mol)
        if len(rings) > 0:
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
        return bonds

    def fragmenize(self, mol, dummyStart=1):
        rings = get_rings(mol)
        if len(rings) > 0:
            bonds = []
            for ring in rings:
                rest_atom_idx = get_other_atom_idx(mol, ring)
                bonds += find_parts_bonds(mol, [rest_atom_idx, ring])
            bonds = self.bonds_filter(mol, bonds)
            if len(bonds) > 0:
                bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
                bond_ids = list(set(bond_ids))
                dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
                break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
                dummyEnd = dummyStart + len(dummyLabels) - 1
            else:
                break_mol = mol
                dummyEnd = dummyStart - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1
        return break_mol, dummyEnd


class BRICS_RING_R_Fragmenizer():
    def __init__(self, break_ring=True):
        self.type = 'BRICS_RING_R_Fragmenizer'
        self.break_ring = break_ring
        self.brics_fragmenizer = BRICS_Fragmenizer()
        self.ring_r_fragmenizer = RING_R_Fragmenizer()

    def fragmenize(self, mol, dummyStart=1):
        brics_bonds = self.brics_fragmenizer.get_bonds(mol)
        if self.break_ring:
            ring_r_bonds = self.ring_r_fragmenizer.get_bonds(mol)
            bonds = brics_bonds + ring_r_bonds
        else:
            bonds = brics_bonds

        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]
            bond_ids = list(set(bond_ids))
            dummyLabels = [(i + dummyStart, i + dummyStart) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            dummyEnd = dummyStart + len(dummyLabels) - 1
        else:
            break_mol = mol
            dummyEnd = dummyStart - 1

        return break_mol, dummyEnd


# 函数一，获取marker邻居原子的index, 注意marker只能是一个单键连接核上的原子，否则邻居会多于一个
def get_neiid_bysymbol(mol, marker):
    try:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == marker:
                neighbors = atom.GetNeighbors()
                if len(neighbors) > 1:
                    print('Cannot process more than one neighbor, will only return one of them')
                atom_nb = neighbors[0]
                return atom_nb.GetIdx()
    except Exception as e:
        print(e)
        return None


# 函数二，获取marker原子的index
def get_id_bysymbol(mol, marker):
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == marker:
            return atom.GetIdx()


def combine2frags(mol_a, mol_b, maker_b='Cs', maker_a='Fr'):
    # 将两个待连接分子置于同一个对象中
    merged_mol = Chem.CombineMols(mol_a, mol_b)
    bind_pos_a = get_neiid_bysymbol(merged_mol, maker_a)
    bind_pos_b = get_neiid_bysymbol(merged_mol, maker_b)
    # 转换成可编辑分子，在两个待连接位点之间加入单键连接，特殊情形需要其他键类型的情况较少，需要时再修改
    ed_merged_mol = Chem.EditableMol(merged_mol)
    ed_merged_mol.AddBond(bind_pos_a, bind_pos_b, order=Chem.rdchem.BondType.SINGLE)
    # 将图中多余的marker原子逐个移除，先移除marker a
    marker_a_idx = get_id_bysymbol(merged_mol, maker_a)
    ed_merged_mol.RemoveAtom(marker_a_idx)
    # marker a移除后原子序号变化了，所以又转换为普通分子后再次编辑，移除marker b
    temp_mol = ed_merged_mol.GetMol()
    marker_b_idx = get_id_bysymbol(temp_mol, maker_b)
    ed_merged_mol = Chem.EditableMol(temp_mol)
    ed_merged_mol.RemoveAtom(marker_b_idx)
    final_mol = ed_merged_mol.GetMol()
    return final_mol


def find_all_idx(frags_list):
    pattern = r'\[(\d+)\*?\]'
    t = set()
    for s in frags_list:
        nums = [int(x) for x in re.findall(pattern, s)]
        for n in nums:
            t.add(n)
    t = sorted(list(t))
    return t

def find_all_idx2(frags_list):
    pattern = r'\[(\d+)\*?\]'
    t = {}
    for s in frags_list:
        nums = [int(x) for x in re.findall(pattern, s)]
        for n in nums:
            t[n] = t.get(n, 0) + 1
    return t

def is_substructure(query, reference_frag):
    pattern = r'\[\*\]'  # 匹配 [*] 模式的正则表达式
    reference_frag = re.sub(pattern, '', reference_frag)
    mol1 = Chem.MolFromSmiles(reference_frag)
    mol2 = Chem.MolFromSmiles(query)
    # if mol1 is None or mol2 is None:
    #     return False
    matches = mol2.GetSubstructMatches(mol1)
    if len(matches) > 0:
        return True
    else:
        return False

class GraphNode():
    def __init__(self,
                 smiles=None,
                 breakpoints=None,
                 index=None,):
        self.smiles = smiles
        self.breakpoints = breakpoints
        self.index = index

def check(circle, nodes):
    bp_to_num = {}
    for c in circle:
        for bp in nodes[c].breakpoints:
            bp_to_num[bp] = bp_to_num.get(bp, 0) + 1
    for k, v in bp_to_num.items():
        if v < 2:
            return False
    return True

def find_all_cycle(frags_list):
    # 建图
    graph = {}
    nodes = []
    res = []
    for i, frag in enumerate(frags_list):
        node = GraphNode(smiles=frag,
                         breakpoints=find_all_idx([frag]),
                         index=i)
        nodes.append(node)
        for breakpoint in node.breakpoints:
            if breakpoint in graph:
                if node not in graph[breakpoint]:
                    graph[breakpoint].append(node)
            else:
                graph[breakpoint] = [node]

    all_num_nodes = len(nodes)
    _nodes = nodes.copy()
    # 找闭环
    i = 0
    while nodes:
        if i >= len(nodes):
            break
        all_circle = []
        visited = [0 for _ in range(all_num_nodes)]
        root = nodes[i]

        def dfs(root, circle=[]):
            index = root.index
            circle.append(index)
            if visited[index] == 1:
                # circle_copy = circle.copy()
                # first_index = circle_copy.index(circle_copy[-1])
                # all_circle.append(circle_copy[first_index:-1])
                c = circle.copy()
                c.pop(-1)
                all_circle.append(c)
                return
            breakpoints = root.breakpoints
            visited[index] = 1
            children = []
            for bp in breakpoints:
                children += graph[bp]
            children = list(set(children))
            for child in children:
                dfs(child, circle)
                circle.pop(-1)

        dfs(root, [])
        max_length_circle = None
        max_length = 0
        # if len(all_circle) == 0:
        #     i += 1

        for c in all_circle:
            if len(c) > max_length:
                if check(c, _nodes):
                    max_length = len(c)
                    max_length_circle = c
        if max_length_circle is None:
            i += 1
            continue
        res.append(max_length_circle)
        new_nodes = []
        for node in nodes:
            if node.index not in max_length_circle:
                new_nodes.append(node)
        nodes = [node for node in nodes if node.index not in max_length_circle]
        for k, v in graph.items():
            new_v = []
            for nd in v:
                if nd.index not in max_length_circle:
                    new_v.append(nd)
            graph[k] = new_v
    res = [r for r in res if len(r) != 0]
    return res

def reconstruct_mol(frags_list, generate_mode, reference):
    all_circle_index = find_all_cycle(frags_list)
    # 贪心
    all_circle_index = sorted(all_circle_index, key=lambda x: len(x), reverse=True)
    if len(all_circle_index) == 0 or len(all_circle_index[0]) == 1:
        return 'fail'

    all_smiles = []
    all_frags_list = []
    for circle_index in all_circle_index:
        _frags_list = [frags_list[i] for i in range(len(frags_list)) if i in circle_index]
        if len(_frags_list) == 1:
            if '*' in _frags_list[0]:
                continue
            else:
                all_smiles.append(_frags_list)

    #     t = find_all_idx(_frags_list)
    #     query = _frags_list.pop(0)
    #     pre_len = len(_frags_list)
    #     error = 0
    #     while _frags_list:
    #         try:
    #             for k in t:
    #                 pat = re.escape('[' + str(k) + '*]')
    #                 point = 0
    #                 while point < len(_frags_list):
    #                     frag = _frags_list[point]
    #                     if re.search(re.compile(pat), frag) and re.search(re.compile(pat), query):
    #                         frag = re.sub(pat, '[Cs]', frag)
    #                         query = re.sub(pat, '[Fr]', query)
    #                         mol1 = Chem.MolFromSmiles(frag)
    #                         mol2 = Chem.MolFromSmiles(query)
    #                         query_mol = combine2frags(mol1, mol2, maker_a='Cs', maker_b='Fr')
    #                         query = Chem.MolToSmiles(query_mol, isomericSmiles=True)
    #                         _frags_list.pop(point)
    #                     else:
    #                         point += 1
    #             if len(_frags_list) == pre_len:
    #                 raise ValueError('error!')
    #             pre_len = len(_frags_list)
    #         except:
    #             error = 1
    #             break
    #
    #     if error == 1:
    #         continue
    #     all_smiles.append(query)
    #     all_frags_list.append(_frags_list)

        smi = combine_all_fragmens(_frags_list)
        if smi != 'fail':
            all_smiles.append(smi)

    if generate_mode == 'denovo':
        if len(all_smiles) > 0:
            return all_smiles[0]
    else:
        for smi in all_smiles:
            flag = 1
            for ref in reference:
                if not is_substructure(smi, ref):
                    flag = 0
            if flag == 1: return smi

    return 'fail'

def conect_all_fragmens(frags_list_, ori_frags_list):
    # 先将之前的片段拼接起来
    i = 0
    new_frag_list = []
    pre_frags = []
    while i < len(frags_list_):
        f = frags_list_[i]
        if f in ori_frags_list:
            pre_frags.append(ori_frags_list.pop(ori_frags_list.index(f)))
        else:
            new_frag_list.append(f)
        i += 1

    while pre_frags:
        pre_frag, pre_frags = combine_all_fragmens(pre_frags, ignore_fail=True)
        new_frag_list.insert(0, pre_frag)
    return combine_all_fragmens(new_frag_list)

def combine_all_fragmens(frags_list_, ignore_fail=False):
    frags_list = copy.deepcopy(frags_list_)
    t = find_all_idx(frags_list)
    query = frags_list.pop(0)
    while True:
        for k in t:
            pat = re.escape('[' + str(k) + '*]')
            point = 0
            while point < len(frags_list):
                frag = frags_list[point]
                if re.search(re.compile(pat), frag) and re.search(re.compile(pat), query):
                    _frag = re.sub(pat, '[Cs]', frag)
                    _query = re.sub(pat, '[Fr]', query)
                    mol1 = Chem.MolFromSmiles(_frag)
                    mol2 = Chem.MolFromSmiles(_query)
                    query_mol = combine2frags(mol1, mol2, maker_a='Cs', maker_b='Fr')
                    query = Chem.MolToSmiles(query_mol, isomericSmiles=True)
                    frags_list.pop(point)
                else:
                    point += 1

        # success
        if '*' not in query and len(frags_list) == 0:
            return query

        # fail
        # query 已经没有断点了
        if '*' not in query and len(frags_list) != 0:
            if ignore_fail:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return 'fail'
        if '*' in query and len(frags_list) == 0:
            if ignore_fail:
                return query, frags_list
            return 'fail'
        # frags_list中没有断点或者没有与query相同的断点
        cur_t = find_all_idx([query])
        residul_t = find_all_idx(frags_list)
        if len(set(cur_t) & set(residul_t)) == 0:
            if ignore_fail:
                return query, frags_list
            # print(query, frags_list)
            return 'fail'

# def combine_all_fragmens(frags_list_):
#     frags_list = copy.deepcopy(frags_list_)
#     t = find_all_idx(frags_list)
#     query = frags_list.pop(0)
#     flag = 'success'
#     while frags_list:
#         for k in t:
#             pat = re.escape('[' + str(k) + '*]')
#             point = 0
#             while point < len(frags_list):
#                 frag = frags_list[point]
#                 if re.search(re.compile(pat), frag) and re.search(re.compile(pat), query):
#                     frag = re.sub(pat, '[Cs]', frag)
#                     query = re.sub(pat, '[Fr]', query)
#                     mol1 = Chem.MolFromSmiles(frag)
#                     mol2 = Chem.MolFromSmiles(query)
#                     query_mol = combine2frags(mol1, mol2, maker_a='Cs', maker_b='Fr')
#                     query = Chem.MolToSmiles(query_mol, isomericSmiles=True)
#                     frags_list.pop(point)
#                 else:
#                     point += 1
#         # success
#         if '*' not in query and len(frags_list) == 0:
#             break
#         # query 已经没有断点了
#         if '*' not in query and len(frags_list) != 0:
#             flag = 'fail'
#             # break
#         if '*' in query and len(frags_list) == 0:
#             flag = 'fail'
#         # frags_list中没有断点或者没有与query相同的断点
#         cur_t = find_all_idx([query])
#         residul_t = find_all_idx(frags_list)
#         if len(set(cur_t) & set(residul_t)) == 0:
#             flag = 'fail'
#         if flag == 'fail':
#             return flag
#
#     return query
