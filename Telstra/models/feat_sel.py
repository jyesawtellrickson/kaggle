"""
Feature Selection file

Chooses the best features to use in the model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import data_prep
import feat_gen


def feat_sel():
    print("Reading data...")
    # Attempt to open pandas and if fail rerun dataSprep and feat_gen
    try:
        all_df = pd.read_csv('../input/all_df_gen.csv', index_col='id')
    except OSError:
        data_prep.data_prep()
        feat_gen.feat_gen()
        all_df = pd.read_csv('../input/all_df_gen.csv', index_col='id')

    """
    ###################### Feature Selection ############################
    """
    print("Selecting features...")
    # Initially remove known bad features from previous runs
    #all_df = all_df.drop(['event_type_int','event_type_total', 'feature_spread'],axis=1)
    all_df = all_df.drop(['k_means_cat_1'], axis=1)
    ##### Should really be training with other columns included, just not iterated over
    # Find out the features that can be excluded from the removal process
    exclude_feat = ['log_feature_','event_type_','resource_type_','severity_type_']
    exclude_feat.append('defence_2')
    exclude_feat_num = [386, 54, 10, 5]
    include_feat = all_df.columns.tolist()
    for idx, feat in enumerate(exclude_feat):
        print(feat)
        exclude_feat_idx = []
        for i in range(1, exclude_feat_num[idx]+1):
            exclude_feat_idx.append(feat+str(i))
        include_feat = list(set(include_feat) - set(exclude_feat_idx))
    # also remove the other columns
    include_feat = list(set(include_feat)-{'train/test', 'fault_severity'})

    # Remove known *good* features from testing
    # so that the model is not run on these
    good_feat = ['location', 'n_inst_all', 'resource_1st', 'log_feature_total',
                 'defence_3', 'resource_1st','defence_1', 'feature_unique',
                 'feat_1st', 'feat_2nd', 'feat_3rd', 'event_type_int']
    include_feat = list(set(include_feat)-set(good_feat))

    # prepare data for testing
    train_df = all_df[all_df['train/test'] == 'train'].drop(['train/test', 'fault_severity'], axis=1)
    y_train = all_df[all_df['train/test'] == 'train']['fault_severity'].values.astype(int)

    # original eta = 0.3, max_depth = 6
    # Set xgboost paramaters
    param = {'eta': 0.3, 'silent': 1, 'objective': 'multi:softprob', 'max_depth': 8, 'eval_metric': 'mlogloss'}
    param['nthread'] = 4
    param['num_class'] = 3
    param_list = list(param.items())
    num_round = 100
    #
    dropped_feats = []
    worse_feat = 'bananas'
    while worse_feat != '':
        # Train model
        # this is effectively early stopping as you're finding the best round from cv
        best_cvs = []
        # iterate through available features, try removing one and running model
        print("Testing ", len(include_feat), " features.")
        for feat in include_feat:
            X_test = train_df.drop(feat,axis=1).values
            dtrain = xgb.DMatrix(X_test, label=y_train)
            cv_error = xgb.cv(param, dtrain, num_round, nfold=5, early_stopping_rounds=10,seed=0)
            best_cvs.append(cv_error['test-mlogloss-mean'][-1:].values.tolist()[0])
        # Get original cv for comparison
        dtrain = xgb.DMatrix(train_df.values, label=y_train)
        cv_error = xgb.cv(param, dtrain, num_round, nfold=5, early_stopping_rounds=10,seed=0)
        original_cv = cv_error['test-mlogloss-mean'][-1:].values.tolist()[0]
        # Print the cv output for the user
        print("Original CV: ", original_cv)
        for row in zip(best_cvs, include_feat):
            print(row)
        # Next print all the values which show an improvement in cv
        print("")
        cv_improve = original_cv
        worse_feat = ''
        for idx, cv_val in enumerate(best_cvs):
            if cv_val < original_cv:
                print(cv_val, include_feat[idx])
                if cv_val < cv_improve:
                    cv_improve = cv_val
                    worse_feat = include_feat[idx]
        # Search through the ones which can be removed and choose the lowest cv
        if worse_feat != '':
            # Delete the feature with the best savings
            train_df = train_df.drop(worse_feat, axis=1)
            dropped_feats.append(worse_feat)
            include_feat = list(set(include_feat)-set([worse_feat]))
            print("Feature: ", worse_feat, " removed.")
        # continue while cv is decreasing
    all_df = all_df.drop(dropped_feats, axis=1)
    print("Features cleaned: ", dropped_feats)

    print("Writing data...")
    all_df.to_csv('../input/all_df_sel.csv')

    print("Done.")
    return

