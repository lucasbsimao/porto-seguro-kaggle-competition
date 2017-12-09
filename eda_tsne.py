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
n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
s = 200000 #desired sample size
skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
train = pd.read_csv(filename, skiprows=skip)

def scale_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()#feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def feature_eng(data):
    #sum ind bins
    data['ps_ind_sum_bin'] = np.zeros(data.shape[0])
    dcol = [c for c in data.columns if c not in ['id','target']]
    for c in dcol:
        if '_bin' in c and 'ps_ind_' in c: #standard arithmetic
            data['ps_ind_sum_bin'] += data[c]

    data['ps_sqrt(car_15)*reg_03'] = data['ps_reg_03']*np.sqrt(data['ps_car_15']) 
    data['ps_sqrt(car_15)*reg_02'] = data['ps_reg_02']*np.sqrt(data['ps_car_15']) 
    data['ps_sqrt(car_13)*reg_02'] = np.sqrt(data['ps_car_13']) * data['ps_reg_02']
    data['ps_sqrt(sum_reg)'] = np.sqrt(1+data['ps_reg_03']+data['ps_reg_02']+data['ps_reg_01'])
    # data['ps_car_sqrt(13+15)'] = np.sqrt(data['ps_car_13']+data['ps_car_15'])
    data['ps_car_sqrt(13+15)/reg_01'] = np.sqrt(data['ps_car_13']+data['ps_car_15'])*np.sqrt(data['ps_reg_01'])

    # data['ps_car_01_cat'],_ = scale_data(data['ps_car_01_cat'].reshape(-1, 1))
    # data['ps_car_04_cat'],_ = scale_data(data['ps_car_06_cat'].reshape(-1, 1))
    # data['ps_car_11_cat'],_ = scale_data(data['ps_car_11_cat'].reshape(-1, 1))
    # data['ps_ind_01'],_ = scale_data(data['ps_ind_01'].reshape(-1, 1))
    # data['ps_ind_03'],_ = scale_data(data['ps_ind_03'].reshape(-1, 1))
    # data['ps_ind_15'],_ = scale_data(data['ps_ind_15'].reshape(-1, 1))
    # data['ps_car_11'],_ = scale_data(data['ps_car_11'].reshape(-1, 1))
    # data['ps_calc_04'],_ = scale_data(data['ps_calc_04'].reshape(-1, 1))
    # data['ps_calc_05'],_ = scale_data(data['ps_calc_05'].reshape(-1, 1))
    # data['ps_calc_06'],_ = scale_data(data['ps_calc_06'].reshape(-1, 1))
    # data['ps_calc_07'],_ = scale_data(data['ps_calc_07'].reshape(-1, 1))
    # data['ps_calc_08'],_ = scale_data(data['ps_calc_08'].reshape(-1, 1))
    # data['ps_calc_09'],_ = scale_data(data['ps_calc_09'].reshape(-1, 1))
    # data['ps_calc_10'],_ = scale_data(data['ps_calc_10'].reshape(-1, 1))
    # data['ps_calc_11'],_ = scale_data(data['ps_calc_11'].reshape(-1, 1))
    # data['ps_calc_12'],_ = scale_data(data['ps_calc_12'].reshape(-1, 1))
    # data['ps_calc_13'],_ = scale_data(data['ps_calc_13'].reshape(-1, 1))
    # data['ps_calc_14'],_ = scale_data(data['ps_calc_14'].reshape(-1, 1))

    data = data.drop(['id',
                'ps_ind_10_bin', 
                'ps_ind_11_bin', 
                'ps_ind_12_bin', 
                'ps_ind_13_bin',
                'ps_car_10_cat',
                'ps_ind_14'],axis=1)

    return data

train = feature_eng(train)

# tsne = TSNE(n_components=2, random_state=1001, perplexity=50, n_iter=1000, verbose=1)
# tsne_results = tsne.fit_transform(train)

#train['x-tsne'] = tsne_results[:,0]
#train['y-tsne'] = tsne_results[:,1]

# colors = ['blue', 'red']
# y = train['target']
# target_names = np.unique(y)

# plt.figure(2, figsize=(10, 10))

# for color, i, target_name in zip(colors, [0, 1], target_names):
#     plt.scatter(tsne_results[y == i, 0], tsne_results[y == i, 1], color=color, s=1,
#                 alpha=.8, label=target_name, marker='.')
# plt.legend(loc='best', shadow=False, scatterpoints=3)
# plt.title('Scatter plot of t-SNE embedding')
# plt.xlabel('X')
# plt.ylabel('Y')

# plt.savefig('t-SNE-porto-01.png', dpi=150)
# plt.show()

colors = train.target
colors = colors*255

#chart = ggplot( train, aes(x='x-tsne', y='y-tsne', fill='target') ) \
chart = ggplot( train, aes(x='ps_sqrt(car_15)*reg_03', y='ps_sqrt(car_13)*reg_02', fill="target") ) \
        + geom_point(color=('lightblue','red') \
        + ggtitle("tSNE dimensions colored by digit")

print(chart)