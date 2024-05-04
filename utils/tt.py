from rdkit import Chem, DataStructs
from rdkit import Chem
import os

def calculate_Tanimoto(mol_a, mol_b):
    fp1 = Chem.RDKFingerprint(mol_a)
    fp2 = Chem.RDKFingerprint(mol_b)
    Tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)
    return Tanimoto

def get_four_score(mol_ligand,mol_genertes):
    tanimoto = []
    for mol_generate in mol_genertes:
        t = calculate_Tanimoto(mol_ligand, mol_generate)
        tanimoto.append(t)
    return sum([1 for i in tanimoto if i==1]),sum([1 for i in tanimoto if i>0.95]),sum([1 for i in tanimoto if i>0.9]),sum([1 for i in tanimoto if i>0.85])

mol_ligand = Chem.MolFromSmiles('CCCC')
mol_genertes = [Chem.MolFromSmiles('C'),Chem.MolFromSmiles('CC'),Chem.MolFromSmiles('CCC'),Chem.MolFromSmiles('CCCC'),Chem.MolFromSmiles('CCCCC')]
print(get_four_score(mol_ligand,mol_genertes))