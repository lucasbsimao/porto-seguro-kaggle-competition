import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import warnings
from collections import Counter
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

print('\nLoading files ...')
filename = "train.csv"
train = pd.read_csv(filename)
filename = "test.csv"
test = pd.read_csv(filename)

id_test = test['id'].values
id_train = train['id'].values
y = train['target']

zeros_like = (train['target'] == 0)*1
ones_like = (train['target'] == 1)*1.5

samples_w = zeros_like+ones_like

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

    data['ps_car_01_cat'],_ = scale_data(data['ps_car_01_cat'].reshape(-1, 1))
    data['ps_car_04_cat'],_ = scale_data(data['ps_car_06_cat'].reshape(-1, 1))
    data['ps_car_11_cat'],_ = scale_data(data['ps_car_11_cat'].reshape(-1, 1))
    data['ps_ind_01'],_ = scale_data(data['ps_ind_01'].reshape(-1, 1))
    data['ps_ind_03'],_ = scale_data(data['ps_ind_03'].reshape(-1, 1))
    data['ps_ind_15'],_ = scale_data(data['ps_ind_15'].reshape(-1, 1))
    data['ps_car_11'],_ = scale_data(data['ps_car_11'].reshape(-1, 1))
    data['ps_calc_04'],_ = scale_data(data['ps_calc_04'].reshape(-1, 1))
    data['ps_calc_05'],_ = scale_data(data['ps_calc_05'].reshape(-1, 1))
    data['ps_calc_06'],_ = scale_data(data['ps_calc_06'].reshape(-1, 1))
    data['ps_calc_07'],_ = scale_data(data['ps_calc_07'].reshape(-1, 1))
    data['ps_calc_08'],_ = scale_data(data['ps_calc_08'].reshape(-1, 1))
    data['ps_calc_09'],_ = scale_data(data['ps_calc_09'].reshape(-1, 1))
    data['ps_calc_10'],_ = scale_data(data['ps_calc_10'].reshape(-1, 1))
    data['ps_calc_11'],_ = scale_data(data['ps_calc_11'].reshape(-1, 1))
    data['ps_calc_12'],_ = scale_data(data['ps_calc_12'].reshape(-1, 1))
    data['ps_calc_13'],_ = scale_data(data['ps_calc_13'].reshape(-1, 1))
    data['ps_calc_14'],_ = scale_data(data['ps_calc_14'].reshape(-1, 1))

    data = data.drop(['id',
                'ps_ind_10_bin', 
                'ps_ind_11_bin', 
                'ps_ind_12_bin', 
                'ps_ind_13_bin',
                'ps_car_10_cat',
                'ps_ind_14'],axis=1)

    return data

train = feature_eng(train.drop(['target'],axis=1))
test = feature_eng(test)
    
#### FEATURES

def eval_gini(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini


def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = -eval_gini(labels, preds)
    return [('gini', gini_score)]

def calculate_model_results(model,name,has_w=False):
    y_valid_pred = 0*y
    y_test_pred = 0

    # Set up folds
    K = 5
    kf = KFold(n_splits = K, random_state = 1, shuffle = True)
    np.random.seed(0)

    for i, (train_index, test_index) in enumerate(kf.split(train)):

        # Create data for this fold
        y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
        X_train, X_valid = train.iloc[train_index,:].copy(), train.iloc[test_index,:].copy()

        print( "\nFold ", i)

        if(not has_w):
            model.fit(X_train, y_train, samples_w[train_index])
        else:
            model.fit(X_train, y_train)

        print("----- Training Done -----")

        pred = model.predict_proba(X_valid)[:,1]
        
        print( "  Gini = ", eval_gini(y_valid, pred) )
        y_valid_pred.iloc[test_index] = pred

        del X_train, X_valid, y_train

    if(not has_w):
        model.fit(train.copy(), y.copy(), samples_w)
    else:
        model.fit(train.copy(), y.copy())
    y_test_pred = model.predict_proba(test.copy())[:,1]
    
    print( "\nGini for full training set:" )
    print(eval_gini(y, y_valid_pred))

    # Create submission file
    sub = pd.DataFrame()
    sub['id'] = id_test
    sub['target'] = y_test_pred
    sub.to_csv('results/'+name+'.csv', float_format='%.6f', index=False)

# train['ps_reg_F'] = train['ps_reg_03'].apply(lambda x: recon(x)[0])
# train['ps_reg_F'],_ = scale_data(train['ps_reg_F'].reshape(-1, 1))

# train['ps_CALC'] = np.sqrt(2+train['ps_reg_F'] +  train['ps_reg_03']) #np.sqrt(14+train['ps_car_12']*train['ps_car_13']*train['ps_car_14']*train['ps_car_15'])

#### TRAINING

algorithm1 = AdaBoostClassifier(n_estimators = 280,learning_rate = 0.7)
algorithm2 = RandomForestClassifier(class_weight='balanced',n_estimators=280, max_depth=4, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
algorithm6 = XGBClassifier(n_estimators=400,
                        max_depth=4,
                        objective="binary:logistic",
                        learning_rate=0.07, 
                        subsample=.8,
                        min_child_weight=6,
                        colsample_bytree=.8,
                        scale_pos_weight=1.6,
                        gamma=10,
                        reg_alpha=8,
                        reg_lambda=1.3,
                        )

# calculate_model_results(algorithm1,"adaboost",has_w=False)
# calculate_model_results(algorithm2,"random_forests",has_w=True)
# calculate_model_results(algorithm6,"xgboost",has_w=True)

# #########################################
# y_valid_pred = 0*y
# y_test_pred = 0

# # Set up folds
# K = 5
# kf = KFold(n_splits = K, random_state = 1, shuffle = True)
# np.random.seed(0)

# for i, (train_index, test_index) in enumerate(kf.split(train)):

#     # Create data for this fold
#     y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
#     X_train, X_valid = train.iloc[train_index,:].copy(), train.iloc[test_index,:].copy()
#     X_test = test.copy()

#     print( "\nFold ", i)

#     algorithm2.fit(X_train, y_train)
#     print("----- Training Done -----")

#     pred = algorithm2.predict_proba(X_valid)[:,1]
#     print( "  Gini = ", eval_gini(y_valid, pred) )
#     y_valid_pred.iloc[test_index] = pred

#     del X_train, X_valid, y_train

# algorithm2.fit(train.copy(), y.copy())
# y_test_pred = algorithm2.predict_proba(test.copy())[:,1]

# print( "\nGini for full training set:" )
# print(eval_gini(y, y_valid_pred))

# # Create submission file
# sub = pd.DataFrame()
# sub['id'] = id_test
# sub['target'] = y_test_pred
# sub.to_csv('random_forests.csv', float_format='%.6f', index=False)

# # #########################################
# y_valid_pred = 0*y
# y_test_pred = 0

# # Set up folds
# K = 5
# kf = KFold(n_splits = K, random_state = 1, shuffle = True)
# np.random.seed(0)

# for i, (train_index, test_index) in enumerate(kf.split(train)):

#     # Create data for this fold
#     y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
#     X_train, X_valid = train.iloc[train_index,:].copy(), train.iloc[test_index,:].copy()
#     X_test = test.copy()

#     print( "\nFold ", i)

#     algorithm3.fit(X_train, y_train, samples_w[train_index])
#     print("----- Training Done -----")

#     pred = algorithm3.predict_proba(X_valid)[:,1]
#     print( "  Gini = ", eval_gini(y_valid, pred) )
#     y_valid_pred.iloc[test_index] = pred

#     del X_train, X_valid, y_train

# algorithm3.fit(train.copy(), y.copy(), samples_w)
# y_test_pred = algorithm3.predict_proba(test.copy())[:,1]

# print( "\nGini for full training set:" )
# print(eval_gini(y, y_valid_pred))

# # Create submission file
# sub = pd.DataFrame()
# sub['id'] = id_test
# sub['target'] = y_test_pred
# sub.to_csv('gbm.csv', float_format='%.6f', index=False)

# #########################################
# y_valid_pred = 0*y
# y_test_pred = 0

# # Set up folds
# K = 5
# kf = KFold(n_splits = K, random_state = 1, shuffle = True)
# np.random.seed(0)

# for i, (train_index, test_index) in enumerate(kf.split(train)):

#     # Create data for this fold
#     y_train, y_valid = y.iloc[train_index].copy(), y.iloc[test_index]
#     X_train, X_valid = train.iloc[train_index,:].copy(), train.iloc[test_index,:].copy()
#     X_test = test.copy()

#     print( "\nFold ", i)

#     algorithm6.fit(X_train, y_train)
#     print("----- Training Done -----")

#     pred = algorithm6.predict_proba(X_valid)[:,1]
#     print( "  Gini = ", eval_gini(y_valid, pred) )
#     y_valid_pred.iloc[test_index] = pred

#     del X_train, X_valid, y_train

# algorithm6.fit(train.copy(), y.copy(), sample_weight=samples_w[train_index])
# y_test_pred = algorithm6.predict_proba(test.copy())[:,1]

# print( "\nGini for full training set:" )
# print(eval_gini(y, y_valid_pred))

# # Create submission file
# sub = pd.DataFrame()
# sub['id'] = id_test
# sub['target'] = y_test_pred
# sub.to_csv('xgb_submit.csv', float_format='%.6f', index=False)