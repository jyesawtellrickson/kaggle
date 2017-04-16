# util.py
# ----------
#
# Script for my first model

import pandas as pd
import numpy as np
import datetime

def string_to_days(myString, date_format=None):
    """
    Take a date in a string format and then return the days from
    1970-01-01 in the integer format
    """
    import datetime
    from datetime import datetime
    if not isinstance(myString, str):
        return None
    myDate = datetime.strptime(myString,'%Y-%m-%d')
    refDate = datetime.strptime('2010-01-01','%Y-%m-%d')
    seconds = (myDate - refDate).total_seconds()
    days = seconds / (60 * 60 * 24)
    days = round(days)
    return days


def get_category_averages(dataFrame, categories):
    """
    This function should deal with cases where averages are small
    """

    category_options = {}
    category_dimensions  = []
    
    for category in categories:
        # Get the unique category options and put them in a dictionary
        category_options[category] = np.unique(dataFrame[category]).tolist()
        # Create a list with the dimensions of the categories
        category_dimensions.append(len(category_options[category]))

    # Create the averages array with the dimensions
    category_averages = np.zeros(category_dimensions)

    # Get averages (only works for 2D)
    for i in range(0,category_dimensions[0]):
        for j in range(0,category_dimensions[1]):
            category_mask = (dataFrame[categories[0]] == \
                             category_options[categories[0]][i]) & \
                             (dataFrame[categories[1]] == \
                              category_options[categories[1]][j])
            category_averages[i,j] = dataFrame[category_mask]['age'].dropna().median()
            if np.nan_to_num(category_averages[i,j]) == 0.:
                category_averages[i,j] = dataFrame['age'].dropna().median()

    category_averages = np.nan_to_num(category_averages)
    
    return category_averages

def category_to_int(dataFrame, dataLabel):
    """
    Takes a dataFrame and dataLabel and converts the labels to integers.
    """
    data_labels = list(enumerate(dataFrame[dataLabel].unique()))
    data_labels_dict = { name : i for i, name in data_labels}
    newDataLabel = dataLabel+'_int'
    dataFrame[newDataLabel] = dataFrame[dataLabel]
    dataFrame[newDataLabel] = dataFrame[newDataLabel] \
                                .map( lambda x: data_labels_dict[x]).astype(int)
    train = dataFrame.drop(dataLabel,axis=1)

    return


def aggregate_session_data():
    session = pd.read_csv('Data/sessions.csv',header=0)
    session['id'] = session['user_id']
    session.drop('user_id',axis=1)
    session_ag = pd.DataFrame({'id':np.unique(session['id'])})
    session_ag['primary_device'] = None
    session_ag['number_devices'] = None
    session_ag['time_elapsed'] = 0
    number_ids = session_ag['id'].size
    session_drop = session
    del session
    for idx, idp in enumerate(session_ag['id']):
        # Create a mask for accessing the specific person's data
        # This is most time consuming
        session_mask = session_drop['id']==idp
        # Find the devices and number of uses by the person
        idp_devices = np.unique(session_drop.loc[session_mask,'device_type'] \
                                ,return_counts=True)
        # Create a mask for the aggregate array
        session_ag_mask = session_ag['id']==idp
        # Write the primary device and number of devices to the aggeregate
        session_ag.loc[session_ag_mask,'primary_device'] = idp_devices[0] \
                                                                 [idp_devices[1].argmax()]
        session_ag.loc[session_ag_mask,'number_devices'] = idp_devices[0].size
        # Find the total time spent by the user on airbnb
        idp_time_elapsed = np.sum(session_drop.loc[session_mask,'secs_elapsed'])
        # Write this to the aggregate matrix
        session_ag.loc[session_ag_mask,'time_elapsed'] = idp_time_elapsed
        # All the user data has been found, remove from matrix for quicker action
        #session_drop = session_drop[np.invert(session_mask)]
        # Print progress to the user
        print(idx/number_ids*100,idp_time_elapsed, idp_devices)
        











    
    

