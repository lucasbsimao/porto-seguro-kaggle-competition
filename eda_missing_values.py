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
import missingno as msno

print('\nLoading files ...')
filename = "train.csv"
train = pd.read_csv(filename)


train = train.drop(['id'], axis=1).reset_index(drop=True)

total_rows = train.shape[0]
total_coluns = train.shape[1]

train = train.replace(-1, np.NaN)

# Nullity or missing values by columns
msno.matrix(df=train.iloc[:,2:50], figsize=(20, 14), color=(0.35, 0.2, 0.2))

missing_values = train.isnull().sum().sort_values()/total_rows

train_mean = train.dropna().mean()
train_std = train.dropna().std()

print(missing_values)

print(train_mean['ps_car_14'])
print(train_mean['ps_reg_03'])
print(train_mean['ps_car_05_cat'])
print(train_mean['ps_car_03_cat'])
print("STD")
print(train_std['ps_car_14'])
print(train_std['ps_reg_03'])
print(train_std['ps_car_05_cat'])
print(train_std['ps_car_03_cat'])

def replace_values(mean,std,missing):
    min = mean - std
    max = mean + std
    rep_values = np.zeros(missing.shape[0])
    for p in range(0,missing.shape[0]):
        rep_values[p] = random.uniform(min,max)
    return pd.Series(rep_values)

count_missing= train.isnull()
train['ps_car_03_cat'].fillna(value=replace_values(train_mean['ps_car_03_cat'],train_std['ps_car_03_cat'],count_missing['ps_car_03_cat']))
train['ps_car_05_cat'].fillna(value=replace_values(train_mean['ps_car_05_cat'],train_std['ps_car_05_cat'],count_missing['ps_car_05_cat']))

fig, axis = plt.subplots(2,1)
sns.countplot(x='ps_car_03_cat', data=train, ax=axis[1], order = train['ps_car_03_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_05_cat', data=train, ax=axis[0], order = train['ps_car_05_cat'].value_counts().index.sort_values())

plt.show()