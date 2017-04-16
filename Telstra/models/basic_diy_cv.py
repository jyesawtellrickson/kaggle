"""Basic model for first attempt.

Uses K-best feature selection.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib
import random
from math import log
import csv

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
severity_df = pd.read_csv('../input/severity_type.csv')
resource_df = pd.read_csv('../input/resource_type.csv')
feature_df = pd.read_csv('../input/log_feature.csv')
event_df = pd.read_csv('../input/event_type.csv')

n_train = train_df.shape[0]

#################### Initial cleanup ############################

# Turn severity types into numbers
train_df['location'] = train_df['location'].map(lambda x: int(x.strip("location ")))
test_df['location'] = test_df['location'].map(lambda x: int(x.strip("location ")))
test_df['fault_severity'] = 'target'

# Resource runs from 1-10 possible options
resource_df['resource_type'] = resource_df['resource_type'].map(lambda x: int(x.strip("resource_type ")))
resource_df = pd.get_dummies(resource_df, columns=['resource_type'])
resource_df = resource_df.groupby('id',as_index=False).aggregate(np.sum)
resource_df['resource_type_total'] = resource_df.ix[:, resource_df.columns != 'id'].sum(axis=1)

# Feature LOTS of options
feature_df['log_feature'] = feature_df['log_feature'].map(lambda x: int(x.strip("feature ")))
feature_df = pd.get_dummies(feature_df, columns=['log_feature'])
feature_df = feature_df.groupby('id',as_index=False).aggregate(np.sum)
feature_df['feature_type_total'] = feature_df.ix[:, feature_df.columns != 'id'].sum(axis=1)

# Events 50 options
event_df['event_type'] = event_df['event_type'].map(lambda x: int(x.strip("event_type ")))
event_df = pd.get_dummies(event_df, columns=['event_type'])
event_df = event_df.groupby('id',as_index=False).aggregate(np.sum)
event_df['event_type_total'] = event_df.ix[:, event_df.columns != 'id'].sum(axis=1)

# convert severity count into categories 1-5
severity_df['severity_type'] = severity_df['severity_type'].map(lambda x: int(x.strip("severity_type ")))
severity_df = pd.get_dummies(severity_df, columns=['severity_type'])
# should realy convert float to int

# use the time that the incidents were generated to create features
all_df = pd.concat([train_df, test_df])
all_df['train/test'] = 'train'
all_df['train/test'][n_train:] = 'test'

merged_df = pd.merge(feature_df, resource_df, how='outer', on='id')
merged_df = pd.merge(merged_df, severity_df, how='outer', on='id')
merged_df = pd.merge(merged_df, event_df, how='outer', on='id')

all_df = pd.merge(all_df, merged_df, how='outer', on='id')

########################### Feature Generation #########################################
# Number of incindences at location
location_x_id = all_df.groupby('location').id.nunique()
all_df['n_events'] = all_df.location.map(lambda x: location_x_id[x])
all_df['n_ids'] = all_df['resource_type_total']*all_df['event_type_total']*all_df['feature_type_total']*all_df['n_events']


# Location specific features, e.g. most common event at location

# use t-SNE to check locations and times
# common events, 1st, 2nd and 3rd
# combine events df to be unique id


########################### Validation and Refinement #################################
#
#
pct_val_arr = [0.1, 0.2, 0.3]
eta_arr = [0.2, 0.3, 0.4]
depth_arr = [5, 6, 8]
best_params = [0, 0, 0]
best_score = 100
test_results = []
# Use feature selection K-best
#
for pct_val in pct_val_arr:
    for eta in eta_arr:
        for depth in depth_arr:
            X_train = all_df[all_df['train/test'] == 'train'].drop(['train/test', 'fault_severity'], axis=1).values
            #
            # VALIDATION SET
            # Create validation set, start by choosing random values
            # Should really be trying to mimic the testing set, e.g. some locations not trained
            pct_val = pct_val #0.2
            n_train = X_train.shape[0]
            n_val = int(n_train * pct_val)
            val_ind = random.sample(range(0, n_train), n_val)
            train_ind = list(set(range(0, n_train)) - set(val_ind))
            #
            # Define x and y values
            X_val = X_train[val_ind][:]
            X_train = X_train[train_ind][:]
            #
            y_train = all_df[all_df['train/test'] == 'train']['fault_severity'].values
            #
            y_val = y_train[val_ind][:].astype(int)
            y_train = y_train[train_ind][:].astype(int)
            #
            # Generate training data
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            #
            # original eta = 0.3, max_depth = 6
            # Set xgboost paramaters
            param = {'eta': eta, 'silent': 1, 'objective': 'multi:softprob'}
            param['max_depth'] = depth
            param['nthread'] = 4
            param['num_class'] = 3
            param['eval_metric'] = 'mlogloss'
            param = list(param.items())
            #
            eval_list = [(dtrain, 'train'), (dval, 'eval')]
            #
            # Train model
            num_round = 1000
            bst = xgb.train(param, dtrain, num_round, evals=eval_list, early_stopping_rounds=100)
            #
            y_val_pred = bst.predict(dval,ntree_limit=bst.best_ntree_limit)
            #
            print("Best iteration: ",bst.best_iteration)
            print("Logloss error: ",bst.best_score)
            test_results.append(bst.best_score)
            if bst.best_score < best_score:
                best_score = bst.best_score
                best_params[0] = pct_val
                best_params[1] = eta
                best_params[2] = depth


# Implement early stopping to improve the model

######################## Training ##########################
# Get ids of test set
id_test = all_df[all_df['train/test'] == 'test']['id'].values
X_test = all_df[all_df['train/test'] == 'test'].drop(['train/test','fault_severity'], axis=1).values

#####################
X_train = all_df[all_df['train/test'] == 'train'].drop(['train/test', 'fault_severity'], axis=1).values

# Should really be trying to mimic the testing set, e.g. some locations not trained
pct_val = best_params[0]
n_train = X_train.shape[0]
n_val = int(n_train * pct_val)
val_ind = random.sample(range(0, n_train), n_val)
train_ind = list(set(range(0, n_train)) - set(val_ind))

# Define x and y values
X_val = X_train[val_ind][:]
X_train = X_train[train_ind][:]

y_train = all_df[all_df['train/test'] == 'train']['fault_severity'].values

y_val = y_train[val_ind][:].astype(int)
y_train = y_train[train_ind][:].astype(int)


# Generate training data
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

###############



#X_train = all_df[all_df['train/test'] == 'train'].drop(['train/test', 'fault_severity'], axis=1).values
#y_train = all_df[all_df['train/test'] == 'train']['fault_severity'].values
#dtrain = xgb.DMatrix(X_train, label=y_train)

# Set xgboost paramaters
param = {'max_depth': best_params[2], 'eta': best_params[1], 'silent': 1, 'objective': 'multi:softprob'}
param['nthread'] = 4
param['num_class'] = 3
param['eval_metric'] = 'mlogloss'
param = list(param.items())

eval_list = [ (dtrain, 'train'), (dval, 'eval'),]

# Train model
num_round = 2000
bst2 = xgb.train(param, dtrain, num_round, eval_list, early_stopping_rounds=100)

# Predict
y_pred = bst2.predict(dtest, ntree_limit=bst.best_ntree_limit)  # ntree_limit=bst.best_ntree_limit
print("Best iteration: ",bst2.best_iteration)

########################### Post process ################################
#
# Plot results
#xgb.plot_importance(bst)


# Output results
predictions_file = open("../output/submission.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["id", "predict_0","predict_1","predict_2"])
open_file_object.writerows(zip(id_test, y_pred[:,0], y_pred[:,1], y_pred[:,2]))
predictions_file.close()



# convert y_val to same format
#y_val_df = pd.DataFrame(y_val)
#y_val_cat = pd.get_dummies(y_val_df, columns=[0]).values.astype(int)


# Calculate logloss
def logloss(y, yhat):
    logloss_sum = 0
    for N in range(0, y.shape[0]):
        for M in range(0, y.shape[1]):
            p_ij = max(min(yhat[N][M],1-10^(-15)), 10^(-15))
            logloss_sum += y[N][M] * log(p_ij)
    return -logloss_sum/y.shape[0] #N
