import pandas as pd
from rdkit import Chem
import json
import numpy as np

from pubchemfp import GetPubChemFPs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

datas=['herg','cav','nav']

for data in datas:
    print(data)
    df1 = pd.read_csv('../dataset/'+data + '/' + data + '_train.csv')
    df2 = pd.read_csv('../dataset/'+data + '/' + data + '_eval_60.csv')
    df3 = pd.read_csv('../dataset/'+data + '/' + data + '_eval_70.csv')
    df=pd.concat([df1,df2,df3]).reset_index()
    smile_dic={}

    for i,smile in df.SMILES.items():
        print(i)
        mol = Chem.MolFromSmiles(smile)
        fp_path=data+str(i)

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024).ToList()
        fp2 = GetPubChemFPs(mol)
        fp = np.concatenate((fp1, fp2))
        np.save('fps/' + fp_path + '.npy', fp)

        smile_dic[smile]=fp_path

    with open('fps/' + data + '.json', 'w', encoding='utf-8') as f:
        json.dump(smile_dic, f, ensure_ascii=False, indent=4)
