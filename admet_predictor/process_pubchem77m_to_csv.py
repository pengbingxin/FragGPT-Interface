import pandas as pd
import os
import numpy as np
from tqdm import tqdm

def txt2csv():
    txt_path = 'admet_predictor/temp/temp_gen.txt'
    smiles_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines)

    for i, line in tqdm(enumerate(lines), total=len(lines)):
        smiles = line.strip()
        smiles_list.append([smiles, 'test', 1.0])

    smi_df = pd.DataFrame(smiles_list, columns=['smiles', 'group', 'fake_label'])
    smi_df.to_csv(f'admet_predictor/temp/temp.csv', index=False)




def txt2csv2(random_num):
    txt_path = 'admet_predictor/temps/temp_gen'+ str(random_num)+'.txt'
    smiles_list = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    np.random.seed(42)
    np.random.shuffle(lines)

    for i, line in tqdm(enumerate(lines), total=len(lines)):
        smiles = line.strip()
        smiles_list.append([smiles, 'test', 1.0])

    smi_df = pd.DataFrame(smiles_list, columns=['smiles', 'group', 'fake_label'])
    smi_df.to_csv(f'admet_predictor/temps/temp'+ str(random_num)+'.csv', index=False)

