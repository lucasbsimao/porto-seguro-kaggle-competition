import matplotlib.pyplot as plt
from matplotlib.ticker import OldScalarFormatter
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import binom_test

#np.set_printoptions(threshold=np.nan)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

train = pd.read_csv("./train.csv", na_values=[-1,-1.0])
test = pd.read_csv("./test.csv", na_values=[-1,-1.0])

#bins part 1
fig, axis = plt.subplots(2,4)

# for i in range(2):
#     for j in range(4):
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())#.set_powerlimits((0, 1))
        
sns.countplot(x='ps_ind_06_bin', data=train, ax=axis[0,0])
sns.countplot(x='ps_ind_07_bin', data=train, ax=axis[0,1])
sns.countplot(x='ps_ind_08_bin', data=train, ax=axis[0,2])
sns.countplot(x='ps_ind_09_bin', data=train, ax=axis[0,3])
sns.countplot(x='ps_ind_10_bin', data=train, ax=axis[1,0])
sns.countplot(x='ps_ind_11_bin', data=train, ax=axis[1,1])
sns.countplot(x='ps_ind_12_bin', data=train, ax=axis[1,2])
sns.countplot(x='ps_ind_13_bin', data=train, ax=axis[1,3])

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.7)


plt.show()

#bins part 2
fig, axis = plt.subplots(2,5)

# for i in range(2):
#     for j in range(5):
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())#.set_powerlimits((0, 1))
        
sns.countplot(x='ps_ind_16_bin', data=train, ax=axis[0,0])
sns.countplot(x='ps_ind_17_bin', data=train, ax=axis[0,1])
sns.countplot(x='ps_ind_18_bin', data=train, ax=axis[0,2])
sns.countplot(x='ps_calc_15_bin', data=train, ax=axis[0,3])
sns.countplot(x='ps_calc_16_bin', data=train, ax=axis[1,0])
sns.countplot(x='ps_calc_17_bin', data=train, ax=axis[1,1])
sns.countplot(x='ps_calc_18_bin', data=train, ax=axis[1,2])
sns.countplot(x='ps_calc_19_bin', data=train, ax=axis[1,3])
sns.countplot(x='ps_calc_20_bin', data=train, ax=axis[0,4])

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.9)

plt.show()

#categorical part 1
fig, axis = plt.subplots(3,2)

# for i in range(3):
#     for j in range(2):
#         #axis[i,j].set_yscale('log')
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())

temp_train = train.fillna("NA")

sns.countplot(x='ps_ind_02_cat', data=temp_train, ax=axis[0,0], order = temp_train['ps_ind_02_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_ind_04_cat', data=temp_train, ax=axis[0,1], order = temp_train['ps_ind_04_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_ind_05_cat', data=temp_train, ax=axis[1,0], order = temp_train['ps_ind_05_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_01_cat', data=temp_train, ax=axis[1,1], order = temp_train['ps_car_01_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_02_cat', data=temp_train, ax=axis[2,0], order = temp_train['ps_car_02_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_03_cat', data=temp_train, ax=axis[2,1], order = temp_train['ps_car_03_cat'].value_counts().index.sort_values())

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#categorical part 2
fig, axis = plt.subplots(3,2)

# for i in range(3):
#     for j in range(2):
#         #axis[i,j].set_yscale('log')
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())

sns.countplot(x='ps_car_07_cat', data=temp_train, ax=axis[0,0], order = temp_train['ps_car_07_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_08_cat', data=temp_train, ax=axis[0,1], order = temp_train['ps_car_08_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_09_cat', data=temp_train, ax=axis[1,0], order = temp_train['ps_car_09_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_04_cat', data=temp_train, ax=axis[1,1], order = temp_train['ps_car_04_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_05_cat', data=temp_train, ax=axis[2,0], order = temp_train['ps_car_05_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_06_cat', data=temp_train, ax=axis[2,1], order = temp_train['ps_car_06_cat'].value_counts().index.sort_values())

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#categorical part 3
fig, axis = plt.subplots(2,1)

for i in range(2):
    #axis[i].set_yscale('log')
    axis[i].yaxis.set_major_formatter(OldScalarFormatter())

sns.countplot(x='ps_car_10_cat', data=temp_train, ax=axis[0], order = temp_train['ps_car_10_cat'].value_counts().index.sort_values())
sns.countplot(x='ps_car_11_cat', data=temp_train, ax=axis[1], order = temp_train['ps_car_11_cat'].value_counts().index.sort_values())

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#integer part 1
fig, axis = plt.subplots(3,2)

# for i in range(3):
#     for j in range(2):
#         #axis[i,j].set_yscale('log')
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())

sns.countplot(x='ps_ind_01', data=temp_train, ax=axis[0,0], order = temp_train['ps_ind_01'].value_counts().index.sort_values())
sns.countplot(x='ps_ind_03', data=temp_train, ax=axis[0,1], order = temp_train['ps_ind_03'].value_counts().index.sort_values())
sns.countplot(x='ps_ind_14', data=temp_train, ax=axis[1,0], order = temp_train['ps_ind_14'].value_counts().index.sort_values())
sns.countplot(x='ps_ind_15', data=temp_train, ax=axis[1,1], order = temp_train['ps_ind_15'].value_counts().index.sort_values())
sns.countplot(x='ps_car_11', data=temp_train, ax=axis[2,0], order = temp_train['ps_car_11'].value_counts().index.sort_values())

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#integer part 2
fig, axis = plt.subplots(3,4)

# for i in range(3):
#     for j in range(4):
#         #axis[i,j].set_yscale('log')
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())

sns.countplot(x='ps_calc_04', data=temp_train, ax=axis[0,0], order = temp_train['ps_calc_04'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_05', data=temp_train, ax=axis[0,1], order = temp_train['ps_calc_05'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_06', data=temp_train, ax=axis[0,2], order = temp_train['ps_calc_06'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_07', data=temp_train, ax=axis[0,3], order = temp_train['ps_calc_07'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_08', data=temp_train, ax=axis[1,0], order = temp_train['ps_calc_08'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_09', data=temp_train, ax=axis[1,1], order = temp_train['ps_calc_09'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_10', data=temp_train, ax=axis[1,2], order = temp_train['ps_calc_10'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_11', data=temp_train, ax=axis[1,3], order = temp_train['ps_calc_11'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_12', data=temp_train, ax=axis[2,0],color = "blue", order = temp_train['ps_calc_12'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_13', data=temp_train, ax=axis[2,1],color = "blue", order = temp_train['ps_calc_13'].value_counts().index.sort_values())
sns.countplot(x='ps_calc_14', data=temp_train, ax=axis[2,2],color = "blue", order = temp_train['ps_calc_14'].value_counts().index.sort_values())

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#float features part 1

fig, axis = plt.subplots(3,2)

# for i in range(3):
#     for j in range(2):
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())

temp_train = train

axis[0,0].hist(temp_train['ps_reg_01'].dropna())
axis[0,0].set_xlabel('ps_reg_01')
axis[0,1].hist(temp_train['ps_reg_02'].dropna())
axis[0,1].set_xlabel('ps_reg_02')
axis[1,0].hist(temp_train['ps_reg_03'].dropna())
axis[1,0].set_xlabel('ps_reg_03')
axis[1,1].hist(temp_train['ps_calc_01'].dropna())
axis[1,1].set_xlabel('ps_calc_01')
axis[2,0].hist(temp_train['ps_calc_02'].dropna())
axis[2,0].set_xlabel('ps_calc_02')
axis[2,1].hist(temp_train['ps_calc_03'].dropna())
axis[2,1].set_xlabel('ps_calc_03')
#sns.distplot(a=temp_train['ps_reg_01'].dropna(), kde=False, ax=axis[0,0])

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#float features part 2
fig, axis = plt.subplots(2,2)

# for i in range(2):
#     for j in range(2):
#         axis[i,j].yaxis.set_major_formatter(OldScalarFormatter())

axis[0,0].hist(temp_train['ps_car_12'].dropna())
axis[0,0].set_xlabel('ps_car_12')
axis[0,1].hist(temp_train['ps_car_13'].dropna())
axis[0,1].set_xlabel('ps_car_13')
axis[1,0].hist(temp_train['ps_car_14'].dropna())
axis[1,0].set_xlabel('ps_car_14')
axis[1,1].hist(temp_train['ps_car_15'].dropna())
axis[1,1].set_xlabel('ps_car_15')
#sns.distplot(a=temp_train['ps_reg_01'].dropna(), kde=False, ax=axis[0,0])

plt.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.95, hspace=0.65,
                    wspace=0.7)

plt.show()

#target values
fig, axis = plt.subplots(1,1)

sns.countplot(x='target',data=train, ax=axis)

plt.show()