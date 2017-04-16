#!/usr/bin/env python
"""XGBoost model for Telstra competition.

This script utilises a basic XGB model from xgboost.
"""

import pandas as pd
import xgboost as xgb
import csv
import data_prep
import feat_gen
import feat_sel

# Data preparation
# Attempt to open data and if fail rerun everything
try:
    all_df = pd.read_csv('../input/all_df_sel.csv', index_col='id')
except OSError:
    print("Data couldn't be found. Running models")
    data_prep.data_prep()
    feat_gen.feat_gen()
    feat_sel.feat_sel()
    all_df = pd.read_csv('../input/all_df_sel.csv', index_col='id')

"""
########################### Validation and Refinement #################################
"""
#
# VALIDATION SET
# Create validation set
# Should really be trying to mimic the testing set, e.g. some locations not trained
# Can we use visualisation and work out how the set was chosen?
#
# Define x and y values
X_train = all_df[all_df['train/test'] == 'train'].drop(['train/test', 'fault_severity'], axis=1).values
X_test = all_df[all_df['train/test'] == 'test'].drop(['train/test', 'fault_severity'], axis=1).values
y_train = all_df[all_df['train/test'] == 'train']['fault_severity'].values.astype(int)
id_test = all_df[all_df['train/test'] == 'test'].index.values
#
# Generate training data
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)
#
# original eta = 0.3, max_depth = 6
# Set xgboost paramaters
param = {'eta': 0.3, 'silent': 1, 'objective': 'multi:softprob', 'max_depth': 10, 'eval_metric': 'mlogloss'}
param['nthread'] = 4
param['num_class'] = 3
param_list = list(param.items())
num_round = 1000
#
# Train model
# this is effectively early stopping as you're finding the best round from cv
cv_error = xgb.cv(param, dtrain, num_round, nfold=5, early_stopping_rounds=30, seed=0)
print("CV Error: ", cv_error['test-mlogloss-mean'][-1:].values.tolist())
#
num_round = cv_error.shape[0]
bst = xgb.train(param_list, dtrain, num_round)
#
# Predict test results
y_pred = bst.predict(dtest)
#
"""
########################### Post process ################################
"""
#
# Plot results
# xgb.plot_importance(bst)
#
# Are people automatically submitting?
#
# Output results
predictions_file = open("../output/submission.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["id", "predict_0", "predict_1", "predict_2"])
open_file_object.writerows(zip(id_test, y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]))
predictions_file.close()

