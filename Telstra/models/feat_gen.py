"""
    Feature Generation file
"""


def feat_gen():

    import pandas as pd
    import data_prep

    print("Reading in data...")
    # Data preparation
    # Attempt to open pandas and if fail rerun data prep
    try:
        all_df = pd.read_csv('../input/all_df_prep.csv', index_col='id')
    except OSError:
        data_prep.data_prep()
        all_df = pd.read_csv('../input/all_df_prep.csv', index_col='id')
    """
    ################################################ Feature Generation ################################################
    """
    print("Generating features...")
    #
    # Defence is about how much they reacted to the incidence,  low indicates strong reaction
    all_df['defence_1'] = all_df['event_type_total'] / all_df['resource_type_total']
    all_df['defence_2'] = all_df['log_feature_total'] / all_df['resource_type_total']
    all_df['defence_3'] = all_df['feature_unique'] / all_df['resource_type_total']
    # all_df['defence_4'] = all_df['severity_type'] / all_df['resource_type_total']
    #
    print("  artificial")
    # using pairwise addition, subtraction and multiplication of variables which are highly correlated with target
    # mechanically create features, log * +
    # perform correlation test with all features, take the top 5
    # apply the different possible manipulations: + - * / log
    """
    good_feat = ['location', 'n_inst_all']#, 'resource_1st', 'log_feature_total', 'severity_type_int']
    for idx, feat in enumerate(good_feat):
        for idx2, feat2 in enumerate(list(set(good_feat)-set([feat]))):
            all_df['af'+str(idx)+'+'+str(idx2)] = all_df[feat] + all_df[feat2]
            all_df['af'+str(idx)+'-'+str(idx2)] = all_df[feat] - all_df[feat2]
            all_df['af'+str(idx)+'x'+str(idx2)] = all_df[feat] * all_df[feat2]
            all_df['af'+str(idx)+'/'+str(idx2)] = all_df[feat] / all_df[feat2]
    """
    #
    # use t-SNE to check locations and times
    # common events, 1st, 2nd and 3rd

    """
    # Use PCA / dimensional reduction
    for dim in range(3,8):
        k_means_df = pd.DataFrame(np.load('location_categories_'+str(dim)+'.npy'))
        k_means_df.columns = ['id','x','y','K_means_'+str(dim)]
        k_means_df = k_means_df.drop(['x','y'],axis=1)
        k_means_df = pd.get_dummies(k_means_df, columns=['K_means_'+str(dim)])
        all_df = pd.merge(all_df, k_means_df, how='outer', on='id')
        print('merged: ',dim)
    """

    print("  k means")
    k_means_df = pd.read_csv('k_means_loc_2d.csv')
    k_means_df['k_means_cat'] = k_means_df['category']
    k_means_df = k_means_df.drop(['PC1', 'PC2', 'category'], axis=1)
    k_means_df = pd.get_dummies(k_means_df, columns=['k_means_cat'])
    all_df['id'] = all_df.index
    all_df = pd.merge(all_df, k_means_df, how='outer', on='location')
    all_df.index = all_df.id
    all_df = all_df.drop('id', axis=1)

    print("Writing data...")
    all_df.to_csv('../input/all_df_gen.csv')

    print("Done.")
    return

#
#
#
"""
##################################### Tried techniques #############################################################
"""

# conditional probability by dividing by total types
# this should work but causes over-fitting resulting in overall bad performance
"""
total_incidents = all_df.fault_severity.value_counts().sum()
total_train_incidents = all_df.fault_severity.value_counts().sum() - all_df.fault_severity.value_counts()['target']
prob_severity = [0, 0, 0]
for i in range(0, 3):
    prob_severity[i] = all_df.fault_severity.value_counts()[i] / total_train_incidents

#
# Most common event
#
# Ratio between events
# If divisor is 0, then assume average value
all_df['n_inst_0/1'] = (all_df['n_inst_0']+prob_severity[0]) / (all_df['n_inst_1']+prob_severity[1])
all_df['n_inst_0/2'] = (all_df['n_inst_0']+prob_severity[0]) / (all_df['n_inst_2']+prob_severity[2])
all_df['n_inst_1/2'] = (all_df['n_inst_1'] + prob_severity[1]) / (all_df['n_inst_2']+prob_severity[2])

#all_df['']
"""

# counts of each type of severity, -1 as we have added dummy value 'target'
# for each id and location, we want the count of fault_severity 0, 1, 2 that occurred at that location
# for i in all_df.fault_severity.unique().tolist()[:-1]:
#     location_x_id = all_df[all_df['fault_severity'] == i].groupby('location').id.nunique()
#     all_df['n_inst_'+str(i)] = all_df.location.map(lambda x: location_x_id[x] if x in location_x_id else 0)
# Need lots of spaces after for loop
# Total number of instances
# all_df['n_inst'] = all_df['n_inst_0'] + all_df['n_inst_1'] + all_df['n_inst_2']

# ids known to cause problems
# use the time that the incidents were generated to create features
# num_id_groups = 100
# all_df['id_group'] = 0
# for i in range(1,num_id_groups-1):
#     all_df.loc[all_df.id > all_df.id.max()/num_id_groups*i, 'id_group'] = i
