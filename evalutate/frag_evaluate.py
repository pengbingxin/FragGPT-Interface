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
def evaluate(csv_path):
    print(f'处理---{csv_path}')
    print('------------------读数据。。。。。')
    data = pd.read_csv(csv_path)
    smi = data['smiles'].values.tolist()
    valid_smi = []
    for s in smi:
        if Chem.MolFromSmiles(s) is not None:
            valid_smi.append(s)
    print('------------------计算valid unique sa qe plogp。。。。。')
    count = len(valid_smi)/len(smi)
    unique = len(set(valid_smi))/len(valid_smi)
    mol=[Chem.MolFromSmiles(s) for s in valid_smi]
    sa=[]
    qe=[]
    plogp=[]
    for m in tqdm(mol):
        sa.append(sascorer.calculateScore(m))
        qe.append(qed(m))
        plogp.append(env.penalized_logp(m))
    sa_score = sum(sa)/len(mol)
    qe_score = sum(qe)/len(mol)
    plog_score = sum(plogp)/len(mol)
    print('------------------计算recovery。。。。。')
    recovery = get_recovery(csv_path)
    # return count ,unique, sa_score, qe_score, plog_score
    return count ,unique, sa_score, qe_score, plog_score, recovery


# def get_novelty(generated_molecules_smiles, reference_molecules_smiles):
#     # 去除参考分子中的立体信息并转换为Canonical形式
#     linkers_train_nostereo_canon = set()
#     for smi in set(reference_molecules_smiles):
#         mol = Chem.MolFromSmiles(smi)
#         if mol is not None:
#             Chem.RemoveStereochemistry(mol)
#             mol_no_hydrogen = Chem.RemoveHs(mol)
#             canonical_smi = MolStandardize.canonicalize_tautomer_smiles(Chem.MolToSmiles(mol_no_hydrogen))
#             linkers_train_nostereo_canon.add(canonical_smi)

#     # 计算生成的分子中的新颖性
#     count_novel = 0
#     for res in generated_molecules_smiles:
#         canonical_res = MolStandardize.canonicalize_tautomer_smiles(res)
#         if canonical_res not in linkers_train_nostereo_canon:
#             count_novel += 1

#     return count_novel / len(generated_molecules_smiles) * 100


def get_smiles(reference_molecules_smiles):

    linkers_train_nostereo = []
    for smi in tqdm(list(set(reference_molecules_smiles))):
        mol = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(mol)
        linkers_train_nostereo.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))
        
    linkers_train_canon = []
    for smi in tqdm(list(linkers_train_nostereo)):
        try:
            print(MolStandardize.canonicalize_tautomer_smiles(smi))
            linkers_train_canon.append(MolStandardize.canonicalize_tautomer_smiles(smi))
        except:
            continue

    data = pd.DataFrame(linkers_train_canon,columns=['smiles'])
    data.to_csv('/home/pengbingxin/pbx/fraggpt/train.txt',index=None)


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

def get_novelty2(train_txt = '/home/pengbingxin/pbx/fraggpt/paper/smiles/train.txt',generated_molecules_smiles = []):
    with open(train_txt,'r') as f1:
            reference_molecules_smiles = f1.readlines()
            reference_molecules_smiles = [i.strip() for i in reference_molecules_smiles]
    linkers_train_canon_unique = reference_molecules_smiles
    linkers_train_canon_unique_dit = {}
    for i in linkers_train_canon_unique:
        linkers_train_canon_unique_dit[i] = 1
    count_novel = 0
    for res in tqdm(generated_molecules_smiles):
        if res in linkers_train_canon_unique_dit:
            continue
        else:
            count_novel +=1
    return count_novel/len(generated_molecules_smiles)*100


def get_score(smiles,smiles_list):
    true_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(smiles)))
    for i in smiles_list:
        pred_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(i)))
        if true_mol_smi == pred_mol_smi:
            return 1
    return 0

def get_recovery(csv_path):
    data = pd.read_csv(csv_path)
    '''
    #Index(['in_mols', 'frag_mols', 'fragment', 'smiles'], dtype='object')
    in_mols:真实分子
    '''
    grouped = data.groupby(['in_mols', 'frag_mols'])
    count = 0
    for group_name, group_data in grouped:
        in_mols = group_name[0]
        smiles = group_data['smiles'].values.tolist()
        score = get_score(in_mols,smiles)
        if score == 1:
            count +=1
    recovery = count/grouped.ngroups
    return recovery

def get_recovery_rl(smiles,file_path):
    data = pd.read_csv(file_path)

    grouped = data.groupby(['in_mols', 'frag_mols'])
    count = 0
    for group_name, group_data in grouped:
        in_mols = group_name[0]
        smiles = group_data['smiles'].values.tolist()
        score = get_score(in_mols,smiles)
        if score == 1:
            count +=1
    recovery = count/grouped.ngroups
    return recovery


import os
import glob

def find_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files





def get_single_evaluate(train_txt,file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
        smi = data['smiles'].values.tolist()
    if file_path.endswith('.txt'):
        with open(file_path) as f:
            smi = f.readlines()
            smi = [x.strip() for x in smi]
    valid_smi = []
    for s in smi:
        if Chem.MolFromSmiles(s) is not None:
            valid_smi.append(s)
    print('------------------计算recovery。。。。。')
    novelty = get_novelty2(train_txt,valid_smi)
    _ ,unique, sa_score, qe_score, plog_score,recovery = evaluate(file_path)
    count = len(valid_smi)
    data_dict = {
    'name': [file_path],
    'count': [count],
    'unique': [unique],
    'sa_score': [sa_score],
    'qe_score': [qe_score],
    'plog_score': [plog_score],
    'recovery': [recovery],
    'novelty': [novelty],
    }
    return data_dict


def get_sa_qed_plogp(smiles_list):

    mol=[Chem.MolFromSmiles(s) for s in tqdm(smiles_list)]

    sa=[]
    qe=[]
    plogp=[]
    for m in tqdm(mol):
        sa.append(sascorer.calculateScore(m))
        qe.append(qed(m))
        plogp.append(env.penalized_logp(m))
    
    sa_score = sum(sa)/len(mol)
    qe_score = sum(qe)/len(mol)
    plog_score = sum(plogp)/len(mol)
    return sa_score,qe_score,plog_score

if __name__ == '__main__':  
    # import argparse
    # parser = argparse.ArgumentParser(description="GENERATE SAMPLE")
    # parser.add_argument("--file_path", type=str, default='/home/pengbingxin/pbx/fraggpt/outputs/paper_admet/rgroup/fraggpt_rgroup_casf_test_admet_15_44444.csv',
    #                     help="test file path")
    # args = parser.parse_args()
    # train_txt = '/home/pengbingxin/pbx/fraggpt/train.txt'
    # data_dict = get_single_evaluate(train_txt,args.file_path)
    # print(f'file_path:{args.file_path} --data_dict {data_dict}')
    '''
Oc1nc(sc1C(=O)N1CCCO1)-c1cccc(c1)F    
Oc1nc(sc1C(=O)N1CCCCO1)-c1cccc(c1)F    

'''
    m = Chem.MolFromSmiles('Oc1nc(sc1C(=O)N1CCCO1)-c1cccc(c1)F')
    print(sascorer.calculateScore(m))
    print(qed(m))
    print(env.penalized_logp(m))
    print('--------------------------')
    m = Chem.MolFromSmiles('Oc1nc(sc1C(=O)N1CCCCO1)-c1cccc(c1)F')
    print(sascorer.calculateScore(m))
    print(qed(m))
    print(env.penalized_logp(m))

    #admet
    '''
    
    77250
    86.62%
    linker_casf.csv 
    0.8662
    'unique': [0.3772135630706696], 
    'sa_score': [3.9566763749870026], 
    'qe_score': [0.385727140769847], 
    'plog_score': [-0.3964953811451087], 
    'recovery': [0.2677966101694915], 
    'novelty': [98.1648908349149]}
    LogP: 3.0299349353633565
    


    fraggpt_linker_casf_logp.csv
    0.4896 
    'unique': [0.2625304007613408], 
    'sa_score': [4.06729136871798], 
    'qe_score': [0.4664404558746003], 
    'plog_score': [-0.8480498798021145], 
    'recovery': [0.027491408934707903], 
    'novelty': [99.70656656444962]}
     LogP: 2.7896264877400334

    fraggpt_linker_casf_logp_10.csv 
    'count': [63183], 
    'unique': [0.15380719497333142], 
    'sa_score': [3.955042577431468], 
    'qe_score': [0.47904946720986935], 
    'plog_score': [-0.6328344347076942], 
    'recovery': [0.030612244897959183], 
    'novelty': [99.08203155912192]}
    LogP: 2.888251746242222

    fraggpt_linker_casf_logp_12.csv, 
    'count': [56315], 
    'unique': [0.20490100328509278], 
    'sa_score': [3.989699907912198], 
    'qe_score': [0.4737005841839298], 
    'plog_score': [-0.6909397792668204], 
    'recovery': [0.023972602739726026], 
    'novelty': [99.41933765426619]}
    LogP:  2.8673756647472644





    BBB: 0.12417087706193715

    '''










    # linker

    '''
    ============================================linker_zinc
    fraggpt_linker_zinc_10.csv
    count': [97349],
    unique': [0.3559564042773937], 
    'sa_score': [3.0319308911370513], 
    'qe_score': [0.6041763412596068], 
    plog_score': [0.7861140855783808], 
    'recovery': [0.1975], '
    novelty': [99.05083770762924]}


    fraggpt_linker_zinc_12.csv
    count': [96114], 
    'unique': [0.483602805002393],
    'sa_score': [3.0705952474056066], 
    'qe_score': [0.5913081998758714], 
    'plog_score': [0.7811285228174488], 
    'recovery': [0.195], 
    'novelty': [99.15308904009822]}


    fraggpt_linker_zinc_15.csv'
    'count': [91036], 
    'unique': [0.6502921920998287], 
    'sa_score': [3.147045554452743], 
    'qe_score': [0.5635712297020892], 
    'plog_score': [0.7445208826863264], 
    'recovery': [0.2225], 
    'novelty': [99.48262225932598]}

    fraggpt_linker_zinc_18.csv
    'count': [72695], 
    'unique': [0.7788018433179723], 
    'sa_score': [3.258115398172386], 
    'qe_score': [0.5260715688405501], 
    'plog_score': [0.6245368391123464], 
    'recovery': [0.195], 
    'novelty': [99.58869248228902]}

    ------------------------------admet 




    ============================================linker_casf
        77250
    fraggpt_linker_casf_10.csv 
    'count': [69837], 0.9040
    'unique': [0.16842075117774247], 
    'sa_score': [3.888929429580372], 
    'qe_score': [0.4064511137420611], 
    'plog_score': [-0.3852496941420597], 
    'recovery': [0.23809523809523808], 
    'novelty': [97.24787719976517]}

    fraggpt_linker_casf_12.csv, 
    'count': [69417], 0.8986
    'unique': [0.2469279859400435], 
    'sa_score': [3.9055279443949473], 
    'qe_score': [0.40066349955642855], 
    'plog_score': [-0.37443243650504116], 
    'recovery': [0.23389830508474577], 
    'novelty': [97.45019231600328]}

    fraggpt_linker_casf_15.csv
    count': [66845], 0.8653
    'unique': [0.3788316254020495], 
    'sa_score': [3.956189693349558],
    'qe_score': [0.3861677477818052], 
    'plog_score': [-0.4028693143017474], 
    'recovery': [0.2677966101694915], 
    'novelty': [98.07315431221483]}


    fraggpt_linker_casf_18.csv'
    'count': [57072], 0.7387
    'unique': [0.4986508270255116], 
    'sa_score': [4.0465319921290925], 
    'qe_score': [0.36580614582996296], 
    'plog_score': [-0.5025283975725414], 
    'recovery': [0.29152542372881357], 
    'novelty': [98.26885337818896]}




    #-----------------------------------admet
    fraggpt_linker_casf_admet_10.csv
    'count': [58862], 0.7619
    'unique': [0.2594373279874962], 
    'sa_score': [3.9762619261564573], 
    'qe_score': [0.45538185493410144], 
    'plog_score': [-0.8489040597806359], 
    'recovery': [0.10034602076124567], 
    'novelty': [99.71118888247086]}


    fraggpt_linker_casf_admet_12.csv 
    'count': [55521], 0.7187
    'unique': [0.22746348228598187], 
    'sa_score': [3.9977914199875335], 
    'qe_score': [0.44129917007579506], 
    'plog_score': [-0.49051049798131113], 
    'recovery': [0.02054794520547945], 
    'novelty': [99.55872552727797]}

    fraggpt_linker_casf_admet_15.csv'], 
    'count': [38853], 0.5029
    'unique': [0.28638715157130723], 
    'sa_score': [4.110039518824296], 
    'qe_score': [0.4338473128438222], 
    'plog_score': [-0.701973165846], 
    'recovery': [0.020618556701030927], 
    'novelty': [99.63194605307184]} 


    fraggpt_linker_casf_admet_18.csv
    'count': [13566], 0.1756
    'unique': [0.33252248267728146], 
    'sa_score': [4.371615208668274], 
    'qe_score': [0.40924612698234275], 
    'plog_score': [-1.2851586597020404], 
    'recovery': [0.0048543689320388345], 
    'novelty': [99.85994397759103]}


    ============================================linker_pdb
    fraggpt_linker_pdb10.csv, 
    'count': [75821], 
    'unique': [0.26652246739030083], 
    'sa_score': [3.7108566824717832], 
    'qe_score': [0.4205157185748998], 
    'plog_score': [-0.9092260683908558], 
    'recovery': [0.16987179487179488], 
    'novelty': [99.37088669366007]}

    fraggpt_linker_pdb12.csv
    count': [74729],
    'unique': [0.3943315178846231], 
    'sa_score': [3.733657116977062], 
    'qe_score': [0.41419160234739766], 
    'plog_score': [-0.9053123254818485], 
    'recovery': [0.1955128205128205], 
    'novelty': [99.47142340992117]}

    fraggpt_linker_pdb18.csv
    'count': [55194], 
    'unique': [0.7048954596514114], 
    'sa_score': [3.8496672429990477], 
    'qe_score': [0.3799778098336935], 
    'plog_score': [-0.9922224661031525], 
    'recovery': [0.17307692307692307], 
    'novelty': [99.79345581041417]}

    fraggpt_linker_pdb15.csv
    'count': [70066], 
    'unique': [0.5708474866554392], 
    'sa_score': [3.780934388904416], 
    'qe_score': [0.39865479165872314], 
    'plog_score': [-0.9127564805346591], 
    'recovery': [0.16666666666666666], 
    'novelty': [99.61750349670311]}



    '''
    # rgroup
    '''
    fraggpt_rgroup_casf_test_10.csv'], 
    'count': [54185], 
    'unique': [0.24940481683122637], 
    'sa_score': [3.239166162440808], 
    'qe_score': [0.5069113667951467], 
    'plog_score': [-0.27397825366322315], 
    'recovery': [0.3835616438356164], 
    'novelty': [95.20531512411185]}

    fraggpt_rgroup_casf_test_12.csv
    count': [54000], 
    'unique': [0.3572222222222222], 
    'sa_score': [3.2610877353022247], 
    'qe_score': [0.49999884338793765], 
    'plog_score': [-0.2675002025943756], 
    'recovery': [0.410958904109589], 
    'novelty': [95.7611111111111]}

    fraggpt_rgroup_casf_test_15.csv
    'count': [52578], 
    'unique': [0.5156148959640915], 
    'sa_score': [3.319550408882007], 
    'qe_score': [0.4800162980175677], 
    'plog_score': [-0.27573295856035146], 
    'recovery': [0.4072398190045249], 
    'novelty': [96.4547909772148]}

    fraggpt_rgroup_casf_test_18.csv 
    'count': [45695], 
    'unique': [0.6355181092023198], 
    'sa_score': [3.3900160951218106], 
    'qe_score': [0.4612149777751285], 
    'plog_score': [-0.31218747570917643], 
    'recovery': [0.35585585585585583], 
    'novelty': [97.0500054710581]}


    -----------------------------------------


    rg_casf_15.csv, 
    'count': [40304], 
    'unique': [0.3962137753076618], 
    'sa_score': [3.3498103908649433], 
    'qe_score': [0.5403809464028645], 
    'plog_score': [-0.6914222502089619], 
    'recovery': [0.49099099099099097], 
    'novelty': [93.72518856689163]}

    rg_casf_12.csv'
    'count': [51273], 
    'unique': [0.2887289606615568], 
    'sa_score': [3.3039804570605766], 
    'qe_score': [0.5415473674929605], 
    'plog_score': [-0.5525250089478999], 
    'recovery': [0.5315315315315315], 
    'novelty': [93.25570963275018]}


    #=============================pdb
    fraggpt_rgroup_pdb_test_10.csv 
    'count': [71014], 
    'unique': [0.27219984791731205], 
    'sa_score': [3.4622376722483135], 
    'qe_score': [0.46094115130595986], 
    'plog_score': [-0.825329818983752], 
    'recovery': [0.15734265734265734], 
    'novelty': [98.51719379277326]}

    fraggpt_rgroup_pdb_test_12.csv
    count': [70684], 
    'unique': [0.39928413785297945], 
    'sa_score': [3.4825313446388892], 
    'qe_score': [0.45612443730408875], 
    'plog_score': [-0.817442654998477], 
    'recovery': [0.17132867132867133], 
    'novelty': [98.74370437439873]}

    fraggpt_rgroup_pdb_test_15.csv, 
    'count': [67908], 
    'unique': [0.5718766566531189], 
    'sa_score': [3.532142656002957], 
    'qe_score': [0.4417330708760034], 
    'plog_score': [-0.8244323734527113], 
    'recovery': [0.15384615384615385], 
    'novelty': [99.16210166696118]}

    fraggpt_rgroup_pdb_test_18.csv' 
    'count': [57098], 
    'unique': [0.700094574240779], 
    'sa_score': [3.5949155677304265], 
    'qe_score': [0.42620960493406546], 
    'plog_score': [-0.8942486003181777], 
    'recovery': [0.11538461538461539], 
    'novelty': [99.49034992469088]}




    '''

    #-------------------ADMET 
    #---------------------------------Linker



    #---------------------------------rgroup



    '''
    
    
    '''


    # import pandas as pd
    # test = pd.read_csv('/home/pengbingxin/pbx/fraggpt/data/testData/sc_and_side/pdbbind_test.csv')['smiles'].values.tolist()
    # sa_score,qe_score,plog_score = get_sa_qed_plogp(test)
    # print(sa_score,qe_score,plog_score)


    # file_list = glob.glob('/home/pengbingxin/pbx/fraggpt/outputs/paper2/202312182/*.csv')
    # res = []
    # for file_path in file_list:
    #     data_dict = get_single_evaluate(file_path)
    #     res.append(data_dict)
    # data = pd.DataFrame(res)
    # data.to_csv('/home/pengbingxin/pbx/fraggpt/outputs/paper2/202312182/evaluate1.csv',index=None)







    # train_smiles_path  = '/home/pengbingxin/pbx/fraggpt/paper/cmol/S1PR1/train.txt'
    # with open(train_smiles_path) as f:
    #     train_smiles = f.readlines()
    #     train_smiles = [x.strip() for x in train_smiles]
    # reference_molecules_smiles = train_smiles
    # linkers_train_nostereo = []
    # for smi in tqdm(list(set(reference_molecules_smiles))):
    #     mol = Chem.MolFromSmiles(smi)
    #     Chem.RemoveStereochemistry(mol)
    #     linkers_train_nostereo.append(Chem.MolToSmiles(Chem.RemoveHs(mol)))
        
    # linkers_train_canon = []
    # for smi in tqdm(list(linkers_train_nostereo)):
    #     try:
    #         print(MolStandardize.canonicalize_tautomer_smiles(smi))
    #         linkers_train_canon.append(MolStandardize.canonicalize_tautomer_smiles(smi))
    #     except:
    #         continue

    # data = pd.DataFrame(linkers_train_canon,columns=['smiles'])
    # data.to_csv('/home/pengbingxin/pbx/fraggpt/paper/cmol/S1PR1/train_st.csv',index=None)

    '''
    pdb 73750
    casf 59250
    '''
















