# basicModel.py
# ----------
#
# Script for my first model

# Import the required packages
import pandas as pd
import numpy as np
import csv as csv
import sys
import util
import csv

from sklearn.linear_model import LogisticRegression

"""
------------------- Import the data from csv -------------------------------
"""

# Create a list of the different data inputs
file_list = ['train_users_2', 'age_gender_bkts', 'sessions', 'countries', 'test_users_2']

# All the data should be stored in one overall data object
data = {}

# Import the data using pandas
# Note that if you have a header row you should use header=0

print('\nReading in data...\n')
for f in file_list:
    data[f] = pd.read_csv('../input/' + f + '.csv', header=0)

print('Defining variables...\n')
# Define the variables that we can work with
train = data['train_users_2']
test = data['test_users_2']
session = data['sessions']
age = data['age_gender_bkts']
country = data['countries']

# Remove date of first booking
train = train.drop(['date_first_booking'],axis=1)
test = test.drop(['date_first_booking'],axis=1)

# Perform initial session alteration for id
session['id'] = session['user_id']
session = session.drop(['user_id'], axis=1)

# Get answers to be used for training
target = train['country_destination']

# Create a dummy country for the test data so that it can be combined
test['country_destination'] = 'dummy'

# create a dataframe which is a combination of the train and test data
combine = pd.concat([train, test])

# Remove the country_destination which was added to the test data
test = test.drop('country_destination', axis=1)

# Remove our data varaible as everything is setup
del data

"""
########################### Begin Cleanup #######################################
"""
print('Cleaning data...\n')

"""
Remove the nulls

First deal with all the columns which have null values

"""
print('Removing nulls...\n')
null_categories = []
# Output the null columns to the user
if train.isnull().values.any() == True:
    for category in list(combine.columns.values):
        if combine[category].isnull().values.any() == True:
            null_categories.append(category)

print('Null values found in:', null_categories, '\n')

combine['age'] = combine['age'].astype(float)

combine.loc[combine['age'] > 100, 'age'] = None
combine.loc[combine['age'] < 15, 'age'] = None

# To choose the unknown ages, take the average of the signup_method, computer, browser
age_averages = util.get_category_averages(combine, ['first_device_type', 'signup_method'])

# age_averages = np.nan_to_num(age

for i in range(0, age_averages.shape[0]):
    for j in range(0, age_averages.shape[1]):
        age_mask = (combine['age'].isnull()) & \
                   (combine['first_device_type'] == np.unique(combine['first_device_type']).tolist()[i]) & \
                   (combine['signup_method'] == np.unique(combine['signup_method']).tolist()[j])
        combine.loc[age_mask, 'age'] = age_averages[i, j]

combine['age'] = combine['age'].astype(int)

# For first_affiliate_tracked ...



"""
Convert the datatypes which need fixing
"""
print('Convert datatypes...\n')
# Must convert dates from string to number
# combine['date_account_created_int'] = combine['date_account_created'] \
#                                    .map( lambda x : string_to_days(x)).astype(int)
combine['date_account_created_int'] = combine['date_account_created'].map(util.string_to_days)
combine = combine.drop('date_account_created', axis=1)

# Must also convert timestamp


"""
Convert objects to ints
"""

# Maybe we should standardize our data

# Get the column names and the datatypes
combine_headings = combine.columns.values.tolist()
combine_object_headings = []

for head in combine_headings:
    combine_object_headings += [combine[head].dtypes]

"""
# Deal with nulls before converting to ints
for head in combine_headings:
    print("-----")
    print(head)
    print(combine[head].dtypes)
    print(np.unique(combine[head],return_counts=True))

print("-----")
"""

# For all the datatypes that are objects
for idx, combine_heading in enumerate(combine_headings):
    # If it's an object, we need to modify it
    if combine[combine_heading].dtypes == 'object' and 'id' != combine_heading:
        dataFrame = combine
        dataLabel = combine_heading
        data_labels = list(enumerate(dataFrame[dataLabel].unique()))
        data_labels_dict = {name: i for i, name in data_labels}
        newDataLabel = dataLabel + '_int'
        dataFrame[newDataLabel] = dataFrame[dataLabel]
        dataFrame[newDataLabel] = dataFrame[newDataLabel] \
            .map(lambda x: data_labels_dict[x]).astype(int)
        if dataLabel != 'country_destination':
            combine = dataFrame.drop(dataLabel, axis=1)

# Print the results of the object changes to the user



"""
Feature Generation
"""
# We should group the times into blocks around the mean, with standard deviation
# Not necessary for logistic regression, continuous is better.

# May be worth grouping into sections, e.g. holiday seasons (winter/summer)

# Are we going to add our own features, for distance to certain countries?
# Start with the session data
# We can generate some new features based off the session interactions
# Primary Device
# Number of devices used
# Total time elapsed
# primary action
# Let's loop over each individual id

# Create feature of time diff between date account created and timestamp first active

"""
for idx, p in enumerate(combine['id']):
    if idx % 10 == 0:
        print(idx)
    id_mask = session['id'] == p
    # Check there are entries in the session log
    if session[id_mask].size == 0:
        # If there are no entries, go straight on to the next value
        print(idx)
        continue   
    # Access the session data applying id as a mask
    combine.loc[id_mask, 'device'] = session[id_mask]['device_type'][0:1]
    # Get total time on website
    combine.loc[id_mask, 'time_elapse'] = session[id_mask]['secs_elapsed'].sum()
"""

print('Prepare training and test data...\n')
# combine['age'] = combine['age'].astype(int)

# Drop all the values not to be used
combine = combine.drop(['timestamp_first_active', \
                        'date_account_created_int', \
                        'first_affiliate_tracked_int'], axis=1)


# for category in list(combine.columns.values):
#    if (len(np.unique(combine[category])) > 100)&(category!='id'):
#        print('Dropped:',category,'\n')
#        combine = combine.drop(category,axis=1)


# Get the test data back from the combineing data
test = combine[combine['country_destination'] == 'dummy']
test_ids = test['id']
test = test.drop(['country_destination_int', 'country_destination'], axis=1)
test = test.drop(['id'], axis=1)

train = combine[combine['country_destination'] != 'dummy']
train = train.drop(['country_destination', 'country_destination_int'], axis=1)
train = train.drop('id', axis=1)

print('Final training parameters...\n')

print(train.describe(), '\n')

del combine

############################## Train logistic regression ########################

print('Training model...\n')

train_data = train.values
train_answers = target.values

test_data = test.values

# Split data into training and validation
pct_val = 0.1
num_examples = train_answers.size
split_row = int(num_examples * (1 - pct_val))

# Split the data up into training and validation
validate_data = train_data[split_row + 1:num_examples]
train_data = train_data[0:split_row]
validate_answers = train_answers[(split_row + 1):num_examples]
train_answers = train_answers[0:split_row]

"""
for i in range(0,100):
    print(validate_answers.tolist()[i], guesses.tolist()[i])

print(model.coef_)

"""

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train_data, train_answers)

guesses = []
max_i = 15
# don't use size as this includes rows
[test_size, i] = test_data.shape
# Split data up to avoid memory error
for i in range(0, max_i):
    start_point = round(test_size * i / max_i)
    end_point = round(test_size * (i + 1) / max_i)-1
    if i == max_i-1:
        end_point = end_point + 1
    i_guess = random_forest.predict(test_data[start_point:end_point])
    guesses += i_guess.tolist()

# This is a list not a numpy array so need to use other method

# guesses = random_forest.predict(test_data)

# test_accuracy = random_forest.score(validate_data,validate_answers)

# print('Test accuracy of:',test_accuracy,'\n')

print(random_forest.feature_importances_)

# country_df = pd.DataFrame({ \
#    "id": test['id']



# Output the data
predictions_file = open("../output/submission.csv", 'w', newline='')
open_file_object = csv.writer(predictions_file, dialect='excel')
open_file_object.writerow(['id', 'country'])

output = zip(test_ids.values.tolist(), guesses)

open_file_object.writerows(output)
predictions_file.close()
print('Done.')

"""
import matplotlib.pyplot as plt
plt.figure(figsize=1)
"""
