from rdkit import Chem
import sascorer
from rdkit.Chem.QED import qed
import pandas as pd
import environment as env
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import AllChem
import warnings
from rdkit import RDLogger
from tqdm import tqdm
# 禁用 RDKit 警告
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import MolStandardize




def get_novelty(generated_molecules_smiles,reference_molecules_smiles):
    linkers_train_nostereo = []
    for smi in tqdm(list(set(reference_molecules_smiles))):
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(mol)
        linkers_train_nostereo.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))
        
    linkers_train_canon = []
    for smi in tqdm(list(linkers_train_nostereo)):
        try:
            linkers_train_canon.append(MolStandardize.canonicalize_tautomer_smiles(smi))
        except:
            continue

    
    linkers_train_canon_unique = list(set(linkers_train_canon))

    count_novel = 0
    linkers_train_canon_unique_dit = {}
    for i in linkers_train_canon_unique:
        linkers_train_canon_unique_dit[i] = 1

    for res in tqdm(generated_molecules_smiles):
        if res in linkers_train_canon_unique_dit:
            continue
        else:
            count_novel +=1
    return count_novel/len(generated_molecules_smiles)*100



if __name__ == '__main__':  

    train_smiles_path = '/home/pengbingxin/pbx/fraggpt/paper/cmol/S1PR1/train_st.csv'
    train_smiles = pd.read_csv(train_smiles_path)['smiles'].values.tolist()

    gen_smiles_path = '/home/pengbingxin/pbx/fraggpt/outputs/paper2/denovo_S1PR1_10_10k_3.csv'
    gen_smiles = pd.read_csv(gen_smiles_path)['smiles'].values.tolist()
    linkers_train_canon_unique = list(set(train_smiles))
    count_novel = 0
    linkers_train_canon_unique_dit = {}
    for i in linkers_train_canon_unique:
        linkers_train_canon_unique_dit[i] = 1

    for res in tqdm(gen_smiles):
        if res in linkers_train_canon_unique_dit:
            continue
        else:
            count_novel +=1
    novelty =  count_novel/len(gen_smiles)*100
    valid = []
    for s in gen_smiles:
        if Chem.MolFromSmiles(s) is not None:
            valid.append(s)
    unique= len(set(valid))/len(valid)

    valid = len(valid)/10000
    print('gen_smiles_path:',gen_smiles_path)
    print('novelty:',novelty)
    print('valid:',valid)
    print('unique:',unique)













