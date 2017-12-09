from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ggplot import *

import random
import warnings
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FormatStrFormatter

print('\nLoading files ...')
filename = "train.csv"
train = pd.read_csv(filename)

train = train.drop(['id'], axis=1).reset_index(drop=True)

# #sum car features
dcol = [c for c in train.columns if c not in ['id']]
train['reg_03-car_15'] = (train['ps_reg_03'])-train['ps_car_15'] 
train['sqrt(car_13*reg_03)'] = np.sqrt(train['ps_car_13']) * train['ps_reg_02']
train['car_sum_13+15'] = train['ps_car_13']+train['ps_car_15']
#train['ps_car_sum_12+13+14+15'] = (train['ps_car_13']*train['ps_car_15'])/train['ps_reg_03']
train['car_sqrt(13+15)/reg_01'] = np.sqrt(train['ps_car_13']+train['ps_car_15'])*np.sqrt(train['ps_reg_01'])

#plot
fig, axis = plt.subplots(2,2)

temp_train = train

gs = temp_train['car_sqrt(13+15)/reg_01'].dropna().groupby(temp_train['target'])
axis[0,0].set_xlabel('car_sqrt(13+15)/reg_01')
sns.kdeplot(gs.get_group(0), bw = 0.05,ax = axis[0,0],color="red",shade = True)
sns.kdeplot(gs.get_group(1), bw =0.05,ax = axis[0,0],color="blue",shade = True)

gs = temp_train['reg_03-car_15'].dropna().groupby(temp_train['target'])
axis[0,0].set_xlabel('reg_03-car_15')
sns.kdeplot(gs.get_group(0), bw = 0.05,ax = axis[0,1],color="red",shade = True)
sns.kdeplot(gs.get_group(1), bw =0.05,ax = axis[0,1],color="blue",shade = True)

gs = temp_train['sqrt(car_13*reg_03)'].dropna().groupby(temp_train['target'])
axis[0,0].set_xlabel('sqrt(car_13*reg_03)')
sns.kdeplot(gs.get_group(0), bw = 0.05,ax = axis[1,0],color="red",shade = True)
sns.kdeplot(gs.get_group(1), bw =0.05,ax = axis[1,0],color="blue",shade = True)

gs = temp_train['car_sum_13+15'].dropna().groupby(temp_train['target'])
axis[0,0].set_xlabel('car_sum_13+15')
sns.kdeplot(gs.get_group(0), bw = 0.05,ax = axis[1,1],color="red",shade = True)
sns.kdeplot(gs.get_group(1), bw =0.05,ax = axis[1,1],color="blue",shade = True)

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#sum ind bins
train['ps_ind_sum_bin'] = np.zeros(train.shape[0])
dcol = [c for c in train.columns if c not in ['id','target']]
for c in dcol:
    if '_bin' in c and 'ps_ind_' in c: #standard arithmetic
        train['ps_ind_sum_bin'] += train[c]

fig, axis = plt.subplots(1,1)
        
sns.barplot(y='target',x='ps_ind_sum_bin', data=train, ax=axis)

plt.show()

#sum ind bins
train['ps_calc_sum_bin'] = np.zeros(train.shape[0])
dcol = [c for c in train.columns if c not in ['id','target']]
for c in dcol:
    if '_bin' in c and 'ps_calc_' in c: #standard arithmetic
        train['ps_calc_sum_bin'] += train[c]
        print(c)

fig, axis = plt.subplots(1,1)
        
sns.barplot(y='target',x='ps_calc_sum_bin', data=train, ax=axis)

plt.show()

def recon(reg):
    integer = int(np.round((40*reg)**2)) # gives 2060 for our example
    for f in range(28):
        if (integer - f) % 27 == 0:
            F = f
    M = (integer - F)//27
    return F, M

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler()#feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

fig, axis = plt.subplots(1,1)


train['ps_reg_F'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
train['ps_reg_F'],_ = scale_data(train['ps_reg_F'].reshape(-1, 1))

train['ps_CALC'] = train['ps_reg_F']/train['ps_car_13']
train['ps_CALC'] = np.sqrt(2+train['ps_reg_F'] +  train['ps_reg_03'])

temp_train = train

gs = temp_train['ps_CALC'].dropna().groupby(temp_train['target'])
axis.set_xlabel('ps_CALC')
sns.kdeplot(gs.get_group(0), bw = 0.05,ax = axis,color="red",shade = True)
sns.kdeplot(gs.get_group(1), bw =0.05,ax = axis,color="blue",shade = True)

plt.show()