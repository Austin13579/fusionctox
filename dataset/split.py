import pandas as pd
import numpy as np
import argparse
from sklearn.utils import shuffle

parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='herg', help='which dataset')
parser.add_argument('--rs', type=int, default=0, help='which random seed')
args = parser.parse_args()

dataset = args.ds
seed = args.rs
np.random.seed(seed)
print("Dataset: " + dataset + ", random seed: " + str(seed))
df = pd.read_csv(dataset+'/'+dataset + '_train.csv')
df_pos=df[df['Label']==1]
df_neg=df[df['Label']==0]
df_pos = shuffle(df_pos)
df_neg = shuffle(df_neg)


train_p_num,train_n_num = int(0.8*len(df_pos)),int(0.8*len(df_neg))
print('Train neg: ',train_n_num)
print('Valid neg: ',len(df_neg)-train_n_num)
print('Train pos: ',train_p_num)
print('Valid pos: ',len(df_pos)-train_p_num)
train_p_data,valid_p_data = df_pos.iloc[:train_p_num],df_pos.iloc[train_p_num:]
train_n_data,valid_n_data = df_neg.iloc[:train_n_num],df_neg.iloc[train_n_num:]


df_train = pd.concat([train_p_data, train_n_data], axis=0)
df_valid = pd.concat([valid_p_data, valid_n_data], axis=0)


df_train.to_csv('datas/'+dataset + '_train' + str(seed) + '.csv', index=False)
df_valid.to_csv('datas/'+dataset + '_valid' + str(seed) + '.csv', index=False)
