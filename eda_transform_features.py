import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
import seaborn as sns
from matplotlib.ticker import OldScalarFormatter
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
col_to_drop = X.columns[X.columns.str.endswith('_cat')]
col_to_dummify = X.columns[X.columns.str.endswith('_cat')].astype(str).tolist()

def transform_non_linear(df):
    df = df.replace(-1, np.NaN)
    d_median = df.median(axis=0)
    d_mean = df.mean(axis=0)
    df = pd.DataFrame(df)
    dcol = [c for c in df.columns if c not in ['id','target']]
    df['ps_car_15_x_ps_reg_03'] = df['ps_car_15'] * df['ps_reg_03']
    df['negative_one_vals'] = np.sum((df[dcol]==-1).values, axis=1)
    for c in dcol:
        if '_bin' not in c: #standard arithmetic
            df[c+str('_median_range')] = (df[c].values > d_median[c]).astype(np.int)
            df[c+str('_mean_range')] = (df[c].values > d_mean[c]).astype(np.int)
            #df[c+str('_sq')] = np.power(df[c].values,2).astype(np.float32)
            #df[c+str('_sqr')] = np.square(df[c].values).astype(np.float32)
            #df[c+str('_log')] = np.log(np.abs(df[c].values) + 1)
            #df[c+str('_exp')] = np.exp(df[c].values) - 1

        if 'ps_calc_' in c and int(c[8:10]) > 9 and int(c[8:10]) < 15:
            df[c] =  np.log(df[c].values + 2)

        if 'ps_car_04_cat' in c:
            df[c] =  np.log(df[c].values + 2)

    return df

train_test = transform_non_linear(train_test)

def transform_dummy():
    for col in col_to_dummify:
        dummy = pd.get_dummies(train_test[col].astype('category'))
        columns = dummy.columns.astype(str).tolist()
        columns = [col + '_' + w for w in columns]
        dummy.columns = columns
        print(dummy)
        train_test = pd.concat((train_test, dummy), axis=1)

        # col = col_to_dummify[0]
        # dummy = pd.get_dummies(train_test[col].astype('category'))
        # columns = dummy.columns.astype(str).tolist()
        # columns = [col + '_' + w for w in columns]
        # dummy.columns = columns
        # print(dummy[:16])
        # train_test = pd.concat((train_test, dummy), axis=1)

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