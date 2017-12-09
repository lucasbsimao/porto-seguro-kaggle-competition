import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import random
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb

print('\nLoading files ...')
filename = "train.csv"
train = pd.read_csv(filename)


train = train.reset_index(drop=True)

#sum ind bins
train['ps_ind_sum_bin'] = np.zeros(train.shape[0])
dcol = [c for c in train.columns if c not in ['id','target']]
for c in dcol:
    if '_bin' in c and 'ps_ind_' in c: #standard arithmetic
        train['ps_ind_sum_bin'] += train[c]

#sum calc bins
train['ps_calc_sum_bin'] = np.zeros(train.shape[0])
dcol = [c for c in train.columns if c not in ['id','target']]
for c in dcol:
    if '_bin' in c and 'ps_calc_' in c: #standard arithmetic
        train['ps_calc_sum_bin'] += train[c]
        print(c)

train['ps_sqrt(car_15)*reg_03'] = train['ps_reg_03']*np.sqrt(train['ps_car_15']) 
train['ps_sqrt(car_15)*reg_02'] = train['ps_reg_02']*np.sqrt(train['ps_car_15']) 
train['ps_sqrt(car_13)*reg_02'] = np.sqrt(train['ps_car_13']) * train['ps_reg_02']
train['ps_sqrt(sum_reg)'] = np.sqrt(1+train['ps_reg_03']+train['ps_reg_02']+train['ps_reg_01'])
#train['ps_car_sqrt(13+15)'] = np.sqrt(train['ps_car_13']+train['ps_car_15'])
train['ps_car_sqrt(13+15)/reg_01'] = np.sqrt(train['ps_car_13']+train['ps_car_15'])*np.sqrt(train['ps_reg_01'])

# train['ps_reg_F'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
# train['ps_reg_F'],_ = scale_data(train['ps_reg_F'].reshape(-1, 1))

# train['ps_CALC'] = train['ps_reg_F']/train['ps_car_13']
# #train['ps_CALC'] = np.sqrt(2+train['ps_reg_F'] +  train['ps_reg_03'])

zeros_like = (train['target'] == 0)*1
ones_like = (train['target'] == 1)*1.5

samples_w = zeros_like+ones_like
#### FEATURES

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


# train['ps_reg_F'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
# train['ps_reg_F'],_ = scale_data(train['ps_reg_F'].reshape(-1, 1))

# train['ps_CALC'] = np.sqrt(2+train['ps_reg_F'] +  train['ps_reg_03']) #np.sqrt(14+train['ps_car_12']*train['ps_car_13']*train['ps_car_14']*train['ps_car_15'])

#### TRAINING

zeros_like = (train['target'] == 0)*1
ones_like = (train['target'] == 1)*1.5

samples_w = zeros_like+ones_like
print(samples_w)

algorithm = AdaBoostClassifier(n_estimators = 100,learning_rate = 0.75)
algorithm2 = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
algorithm3 = GradientBoostingClassifier(n_estimators=100, max_depth=8, min_samples_leaf=4, max_features=0.2, random_state=0)
algorithm.fit(train.drop(['id', 'target'],axis=1), train.target, sample_weight=samples_w)
algorithm2.fit(train.drop(['id', 'target'],axis=1), train.target, sample_weight=samples_w)
algorithm3.fit(train.drop(['id', 'target'],axis=1), train.target, sample_weight=samples_w)
print("----- Training Done -----")

# Scatter plot 
features = train.drop(['id', 'target'],axis=1).columns.values


importances = (algorithm.feature_importances_ + algorithm2.feature_importances_ + algorithm3.feature_importances_)/3
indices = np.argsort(importances)

print("importances--")
print(importances.sum())

# df = pd.DataFrame(importances, columns=['feature', 'fscore'])

# plt.figure()
# df.plot()
# df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 25))
# plt.gcf().savefig('feature_importance/GBM-1.png')

print(importances)
print("")
print(indices)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices]) ## removed [indices]
plt.xlabel('Relative Importance')

plt.subplots_adjust(top=0.9, bottom=0.10, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)
plt.show()