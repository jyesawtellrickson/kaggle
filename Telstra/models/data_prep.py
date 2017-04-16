"""
    Function for initial data preparation
"""

import pandas as pd
import numpy as np


def data_prep():
    print("Reading data from csv...")
    # Read in data
    train_df = pd.read_csv('../input/train.csv')
    test_df = pd.read_csv('../input/test.csv')
    severity_df = pd.read_csv('../input/severity_type.csv')
    resource_df = pd.read_csv('../input/resource_type.csv')
    feature_df = pd.read_csv('../input/log_feature.csv')
    event_df = pd.read_csv('../input/event_type.csv')

    # Count number of training examples
    n_train = train_df.shape[0]
    """
    #################### Initial cleanup ############################
    """
    print("Manipulating raw data...")
    print("  location")
    # Turn locations into numbers, there are ~1000 locations, so keep as int
    train_df['location'] = train_df['location'].map(lambda x: int(x.strip("location ")))
    test_df['location'] = test_df['location'].map(lambda x: int(x.strip("location ")))
    test_df['fault_severity'] = -1
    # should just remove fault_severity from the picture...

    print("  resource")
    # Resource runs from 1-10 possible options and can have multiple per id
    resource_df['resource_type'] = resource_df['resource_type'].map(lambda x: int(x.strip("resource_type ")))
    resource_df['resource_type_int'] = resource_df['resource_type']
    resource_df = pd.get_dummies(resource_df, columns=['resource_type'])
    # bring together resources to one entry per id
    resource_df = resource_df.groupby('id', as_index=False).aggregate(np.sum)
    # Create total for resources in case this has meaning
    resource_df['resource_type_total'] = resource_df.ix[:, resource_df.columns != 'id'].sum(axis=1)
    # Find the most frequent value for a given id
    resource_df_2 = resource_df.drop(['id', 'resource_type_total', 'resource_type_int'], axis=1)
    arank = resource_df_2.apply(np.argsort, axis=1)
    ranked_cols = resource_df_2.columns.to_series()[arank.values[:, ::-1][:, :1]]
    new_frame = pd.DataFrame(ranked_cols, index=resource_df_2.index)
    new_frame = new_frame.applymap(lambda x: int(x.strip("resource_type_")))
    new_frame.columns = ['resource_1st']
    # add these back to the original frame
    resource_df = pd.concat([resource_df, new_frame], axis=1)
    # clean up excess resources
    del new_frame
    del resource_df_2

    print("  log features")
    # Feature LOTS of options
    feature_df['log_feature'] = feature_df['log_feature'].map(lambda x: int(x.strip("feature ")))
    # create column with the log feature as int
    feature_df['log_feature_int'] = feature_df['log_feature']
    # create secondary df for manipulation
    feature_df_2 = feature_df.drop(['id', 'log_feature_int'], axis=1)
    # convert features to OHE
    feature_df_2 = pd.get_dummies(feature_df_2, columns=['log_feature'])
    # multiply through by the volume
    feature_df_2['volume'] = feature_df_2['volume'].astype(int)
    feature_df_2 = feature_df_2.mul(feature_df_2['volume'], axis=0)
    # remove the volume now
    feature_df_2 = feature_df_2.drop('volume', axis=1)
    # bring the id's into df to join back up
    feature_df_2['id'] = feature_df['id']
    # apply the groupby to get numbers of each event
    feature_df_2 = feature_df_2.groupby('id', as_index=False).aggregate(np.sum)
    feature_df_2['log_feature_total'] = feature_df_2.ix[:, feature_df_2.columns != 'id'].sum(axis=1)
    feature_df = feature_df_2
    # Should calculate the most common features (top 3)
    feature_df_2 = feature_df_2.drop(['id', 'log_feature_total'], axis=1)
    arank = feature_df_2.apply(np.argsort, axis=1)
    ranked_cols = feature_df_2.columns.to_series()[arank.values[:, ::-1][:, :3]]
    new_frame = pd.DataFrame(ranked_cols, index=feature_df_2.index)
    new_frame = new_frame.applymap(lambda x: int(x.strip("log_feature_")))
    new_frame.columns = ['feat_1st', 'feat_2nd', 'feat_3rd']
    # number of different features recorded for id
    feature_df['feature_unique'] = feature_df[feature_df > 0].count(axis=1) - 2  # take 1 for id and log_feature_total
    feature_df['feature_spread'] = feature_df['log_feature_total'] / feature_df['feature_unique']
    # investigate this further!!
    feature_df = pd.concat([feature_df, new_frame], axis=1)
    del feature_df_2

    print("  events")
    # Events 50 options
    event_df['event_type'] = event_df['event_type'].map(lambda x: int(x.strip("event_type ")))
    event_df['event_type_int'] = event_df['event_type']
    # convert to categorical
    event_df = pd.get_dummies(event_df, columns=['event_type'])
    event_df = event_df.groupby('id', as_index=False).aggregate(np.sum)
    event_df['event_type_total'] = event_df.ix[:, event_df.columns != 'id'].sum(axis=1)
    event_df['event_type_total'] = event_df['event_type_total'] - event_df['event_type_int']
    # Calculate most common event
    event_df_2 = event_df.drop(['id', 'event_type_total', 'event_type_int'], axis=1)
    arank = event_df_2.apply(np.argsort, axis=1)
    ranked_cols = event_df_2.columns.to_series()[arank.values[:, ::-1][:, :1]]
    new_frame = pd.DataFrame(ranked_cols, index=event_df_2.index)
    new_frame = new_frame.applymap(lambda x: int(x.strip("event_type_")))
    new_frame.columns = ['1st']
    event_df = pd.concat([event_df, new_frame], axis=1)
    del event_df_2
    del new_frame

    print("  severity")
    # convert severity count into categories 1-5
    severity_df['severity_type'] = severity_df['severity_type'].map(lambda x: int(x.strip("severity_type ")))
    severity_df['severity_type_int'] = severity_df['severity_type']
    severity_df = pd.get_dummies(severity_df, columns=['severity_type'])

    print("Joining data...")
    # Join train and test
    train_df['train/test'] = 'train'
    test_df['train/test'] = 'test'
    all_df = pd.concat([train_df, test_df])
    # Bring all the extra data
    merged_df = pd.merge(feature_df, resource_df, how='outer', on='id')
    merged_df = pd.merge(merged_df, severity_df, how='outer', on='id')
    merged_df = pd.merge(merged_df, event_df, how='outer', on='id')
    # Final stage
    all_df = pd.merge(all_df, merged_df, how='outer', on='id')
    """
    ########################### Feature Generation #########################################
    """
    # Number of incindences at location (including test values)
    location_x_id = all_df.groupby('location').id.nunique()
    all_df['n_inst_all'] = all_df.location.map(lambda x: location_x_id[x])
    # set index to id and drop the column
    all_df.index = all_df['id']
    all_df = all_df.drop('id', axis=1)
    # send data to csv for later use
    print("Writing data...")
    all_df.to_csv('../input/all_df_prep.csv')

    print("Done.")

    return



