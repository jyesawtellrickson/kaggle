"""Feature Selection file

Chooses the best features to use in the model.
"""


def feat_sel():

    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import data_prep
    import feat_gen

    print("Reading data...")
    # Attempt to open pandas and if fail rerun data prep
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
    import matplotlib as plt
    from sklearn.svm import SVC
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.feature_selection import RFECV
    from sklearn.datasets import make_classification

    # Initially remove known bad features
    # all_df = all_df.drop(['id'],axis=1)
    all_df = all_df.drop(['event_type_int','event_type_total', 'feature_spread'],axis=1)

    ##### Should really be training with other columns included, just not iterated ove0
    # Find out the features that can be excluded from the removal process
    exclude_feat = ['log_feature_','event_type_','resource_type_','severity_type_']
    exclude_feat_num = [386, 54, 10, 5]
    include_feat = all_df.columns.tolist()
    all_sel_df = all_df
    all_sel_df['event_type_16'] = 0
    for idx, feat in enumerate(exclude_feat):
        print(feat)
        exclude_feat_idx = []
        for i in range(1, exclude_feat_num[idx]+1):
            exclude_feat_idx.append(feat+str(i))
        all_sel_df = all_sel_df.drop(exclude_feat_idx,axis=1)
        include_feat = list(set(include_feat) - set(exclude_feat_idx))

    sel_cols = all_sel_df.drop(['train/test','fault_severity'],axis=1).columns.tolist()

    X_train = all_sel_df[all_sel_df['train/test'] == 'train'].drop(['train/test', 'fault_severity'], axis=1).values
    y_train = all_sel_df[all_sel_df['train/test'] == 'train']['fault_severity'].values.astype(int)

    train_df = all_df[all_df['train/test'] == 'train'].drop(['train/test', 'fault_severity'], axis=1)

    # original eta = 0.3, max_depth = 6
    # Set xgboost paramaters
    param = {'eta': 0.3, 'silent': 1, 'objective': 'multi:softprob', 'max_depth': 8, 'eval_metric': 'mlogloss'}
    param['nthread'] = 4
    param['num_class'] = 3
    param_list = list(param.items())
    #
    # Train model
    # this is effectively early stopping as you're finding the best round from cv
    num_round = 1000
    best_cvs = []
"""
    for i in range(0, X_train.shape[1]):
        X_train_2 = np.delete(X_train, i, 1)
        dtrain = xgb.DMatrix(X_train_2, label=y_train)
        cv_error = xgb.cv(param, dtrain, num_round, nfold=5, early_stopping_rounds=10,seed=0)
        print("CV Error: ", cv_error['test-mlogloss-mean'][-1:])
        best_cvs.append(cv_error['test-mlogloss-mean'][-1:].values.tolist()[0])
"""
    for feat in include_feat:
        test_df = train_df.drop(feat,axis=1)
        X_test = test_df.values
        dtrain = xgb.DMatrix(X_test, label=y_train)
        cv_error = xgb.cv(param, dtrain, num_round, nfold=5, early_stopping_rounds=10,seed=0)
        # print("CV Error: ", cv_error['test-mlogloss-mean'][-1:])
        best_cvs.append(cv_error['test-mlogloss-mean'][-1:].values.tolist()[0])

    dtrain = xgb.DMatrix(X_train, label=y_train)
    cv_error = xgb.cv(param, dtrain, num_round, nfold=5, early_stopping_rounds=10,seed=0)
    original_cv = cv_error['test-mlogloss-mean'][-1:].values.tolist()[0]

    print("Original CV: ", original_cv)
    for row in zip(best_cvs,sel_cols):
        print(row)

    print("")
    for idx, cv_val in enumerate(best_cvs):
        if cv_val < original_cv:
            print(cv_val, sel_cols[idx])


    print("Writing data...")
    all_df.to_csv('../input/all_df_sel.csv')

    print("Done.")