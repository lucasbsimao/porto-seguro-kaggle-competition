import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
skip = sorted(random.sample(xrange(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
train = pd.read_csv(filename, skiprows=skip)

y = train['target'].values.astype(np.int8)

X = train.drop(['id', 'target'], axis=1)
n_train = X.shape[0]
train_test = X.reset_index(drop=True)

def scale_data(X, scaler=None):
    if not scaler:
        #scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

train_test_scaled, scaler = scale_data(train_test)

train_test = pd.DataFrame(train_test_scaled,columns =train_test.columns.values)

num_positions = train_test.shape[1]
colormap = plt.cm.magma
plt.figure(figsize=[num_positions,num_positions])

plt.title('Pearson correlation of continuous features', y=1.05, size=15)
# sns.heatmap(train_test.corr(),linewidths=0.1,vmax=1.0, square=True, 
#             cmap=colormap, linecolor='white', annot=True)
sns.heatmap(train_test.corr(),linewidths=0.05, mask=np.zeros_like(train_test.corr(), dtype=np.bool), cmap=plt.cm.magma,
            square=True)

plt.subplots_adjust(top=0.995, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
        wspace=0.7)

plt.show()
fig, axis = plt.subplots(2,3)
fig.canvas.draw()
for i in range(2):
    for j in range(3):
        axis[i,j].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        for tick in axis[i,j].get_xticklabels():
            tick.set_rotation(90)

sns.countplot(x='ps_calc_10', data=train_test, ax=axis[0,0],color = "blue", order = train_test['ps_calc_10'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_11', data=train_test, ax=axis[0,1],color = "blue", order = train_test['ps_calc_11'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_12', data=train_test, ax=axis[0,2],color = "blue", order = train_test['ps_calc_12'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_13', data=train_test, ax=axis[1,0],color = "blue", order = train_test['ps_calc_13'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_14', data=train_test, ax=axis[1,1],color = "blue", order = train_test['ps_calc_14'].value_counts().index.sort_values())

plt.subplots_adjust(top=0.998, bottom=0.20, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()