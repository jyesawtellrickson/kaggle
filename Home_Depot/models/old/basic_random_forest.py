# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
import random as rand
import ngram
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import pipeline, grid_search
import time
start_time = time.time()

# import xgboost

# Porter stemmer is recommended

# maybe I could take the average of three different predictions to mimic the real setup
# an improvement would use semantic distance
# spelling corrections? synonym replacement
# other replacements, eg. ps 2 -> ps2, hardward related
# maybe divide number of matches by length of descriptions
# try using different ngrams, matching on 2s or 3s of words etc.
# should we round estimates to only possible answers

# look at difference between predictions and actual, try to fix biggest differences
print("Reading in data...")
df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

num_train = df_train.shape[0]


# What about words which aren't space seperated.
def str_common_word_ngram(str1, str2, n):
    # what happens if length of word is less than size of gram? should return 0
    # use switcher
    if n == 1:
        return sum(int(str2.find(str(str1))>=0) for word in str1.split())
    elif n == 2:
        return sum(int(str2.find(word_ngram)>=0) for word_ngram in ngram.getBigram(str1.split()," "))
    elif n == 3:
        return sum(int(str2.find(word_ngram)>=0) for word_ngram in ngram.getTrigram(str1.split()," "))
    elif n == 4:
        return sum(int(str2.find(word_ngram)>=0) for word_ngram in ngram.getFourgram(str1.split()," "))
    else:
        print("Incorrect n value entered:",n)
        return 0
# What about words which aren't space seperated.
def str_common_word_nterm(str1, str2, n):
    # what happens if length of word is less than size of gram? should return 0
    # use switcher
    if n == 2:
        return sum(int(str2.find(word_ngram)>=0) for word_ngram in ngram.getBiterm(str1.split()," "))
    elif n == 3:
        return sum(int(str2.find(word_ngram)>=0) for word_ngram in ngram.getTriterm(str1.split()," "))
        
# consider using biterms


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

# Import stemmed and corrected language
try:
    df_all = pd.read_csv('../input/df_all_stemmed_corrected.csv')
except OSError:
    print("Stemmed language is missing!")
    exit


#
#
############################# Feature generation ##############################
#
#
print("Generating features...")

# define features for generation
len_feat = ['len_of_query','len_of_title','len_of_description']
len_feat_source = ['search_term','product_title','product_description']

# Create a length of query which descirbes the number of words in the search term
for i in range(0,3):
    df_all[len_feat[i]] = df_all[len_feat_source[i]].map(lambda x:len(x.split())).astype(np.int64)


word_match_feat = ['word_in_title','word_in_description','word_in_attributes_names','word_in_attributes_values']
# using corrected
word_match_c_feat = [s + "_c" for s in word_match_feat]
# used for maximum
word_match_m_feat = [s + "_m" for s in word_match_feat]
# using bigram
word_match_t_feat = [s + "_t" for s in word_match_feat]
word_match_t_c_feat = [s + "_t_c" for s in word_match_feat]
search_feat = ['search_term','product_title','product_description','attributes_names','attributes_values','search_term_corrected']

word_feat_num = len(word_match_feat)
search_feat_num = len(search_feat)
# maybe good to combine all text in one column and search this as this will 
# show if it's the same word being found over and over again vs different words
# found throughout. The former being better as it reingoferces likelihood.

df_all['product_info'] = ""
# Create product_info which amalgamates all the information, tab seperated
for x in range(0,search_feat_num):
    df_all['product_info'] = df_all['product_info'] + df_all[search_feat[x]] + "\t"

print("   single word match")
# Create feature if the search term is in the product title for both normal and corrected
for i in range(0,word_feat_num):
    df_all[word_match_feat[i]] = df_all['product_info'].map(lambda x:str_common_word_ngram(x.split('\t')[0],x.split('\t')[i+1],1))
    df_all[word_match_c_feat[i]] = df_all['product_info'].map(lambda x:str_common_word_ngram(x.split('\t')[word_feat_num+1],x.split('\t')[i+1],1))

# find maximum of the corrected and non-corrected to use.
for i in range(0,word_feat_num):
    df_all[word_match_m_feat[i]] = df_all[[word_match_feat[i],word_match_c_feat[i]]].max(axis=1)

# Remove the corrected features as they are no longer required
# ALTHOUGH they may be useful if you consider that the corrected ones are probably more likely to be a match
df_all = df_all.drop(word_match_c_feat,axis=1)
df_all = df_all.drop(word_match_feat,axis=1)

# Should use relative words. i.e. divide by number of words in title/description

print("   two word match")
# Check if all terms in title
for i in range(0,word_feat_num):
    df_all[word_match_t_feat[i]] = df_all['product_info'].map(lambda x:str_common_word_ngram(x.split('\t')[0],x.split('\t')[i+1],2))
    df_all[word_match_t_c_feat[i]] = df_all['product_info'].map(lambda x:str_common_word_ngram(x.split('\t')[word_feat_num+1],x.split('\t')[i+1],2))

# find maximum of the corrected and non-corrected to use.
for i in range(0,word_feat_num):
    df_all[word_match_t_feat[i]] = df_all[[word_match_t_feat[i],word_match_t_c_feat[i]]].max(axis=1)

df_all = df_all.drop(word_match_t_c_feat,axis=1)


### Bi/Triterms
#
# Check if all terms in title
for i in range(0,word_feat_num):
    df_all[word_match_t_feat[i]] = df_all['product_info'].map(lambda x:str_common_word_nterm(x.split('\t')[0],x.split('\t')[i+1],2))
    df_all[word_match_t_c_feat[i]] = df_all['product_info'].map(lambda x:str_common_word_nterm(x.split('\t')[word_feat_num+1],x.split('\t')[i+1],2))





df_all['total_matches'] = 0
df_all['total_matches_t'] = 0
for i in range(0,word_feat_num):
    df_all['total_matches'] = df_all['total_matches'] + df_all[word_match_m_feat[i]]*3/4 *2 
    df_all['total_matches_t'] = df_all['total_matches'] + df_all[word_match_t_feat[i]]*3/4 * 8  
#((df_all[word_match_t_feat].sum(axis=1))*3/4 ) * 8 / df_all['len_of_query']

#/ df_all['len_of_query']

print("Preparing for predicting...")
# Remove excess features
df_all = df_all.drop(search_feat,axis=1)
df_all = df_all.drop(["product_info","product_uid"],axis=1)
# Remaining: len_of_query, word_in_title, word_in_description
#df_all = df_all.drop(["word_in_attributes_names_m","word_in_attributes_values_m","word_in_attributes_names_t","word_in_attributes_values_t"],axis=1)

# Some features are known to detrimant the algorithm, remove these before
# continuing



df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']


X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

y_train = df_train['relevance'].values

# How many samples we want to calculate cv with
num_cv = int(num_train*0.3)
# get indices to sample from
rand_ind=rand.sample(range(0,num_train),num_cv)
            


RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)
rfr = RandomForestRegressor()
clf = pipeline.Pipeline([('rfr', rfr)])
param_grid = {'rfr__n_estimators' : [300],# 300 top
              'rfr__max_depth': [None], #list(range(7,8,1))
            }
model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 10, scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)
print(len(y_pred))
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_500.csv',index=False)
print("--- Training & Testing: %s minutes ---" % ((time.time() - start_time)/60))

"""
# Try different parameters to optimise result
train_depths = [None]
train_estimators = [12] # [10, 11, 12, 15, 20]
train_features = ["log2",0.1,0.3] #[sqrt]
train_b_estimators = [45, 60]
train_samples = [0.5, 0.7, 0.9]
for train_sample in train_samples:
    for train_b_estimator in train_b_estimators:
        for train_feature in train_features:
            for train_depth in train_depths:
                for train_estimator in train_estimators:
                    print("Learning...")
                    rf = RandomForestRegressor(n_estimators=train_estimator,  \
                        max_depth=train_depth, max_features=train_feature, random_state=None)
                    clf = BaggingRegressor(rf, n_estimators=train_b_estimator, max_samples=train_sample, random_state=None)
                    clf.fit(X_train, y_train)
                        
                    
                    ### Post analysis
                    #
                    print('Checking results...')
                    
                    # predict the results for cv
                    cv_pred = clf.predict(X_train[rand_ind])
                    # estimate the rmse
                    rmse_est = np.sqrt(sum((cv_pred - y_train[rand_ind])**2)/num_cv)
                    
                    print("Estimated RMSE = ",rmse_est)
                    print("Paramters: bagging: ",train_b_estimator,"  sample: ",train_sample, "  feature: ",train_feature)






### Predict the results
#
print("Predicting...")
y_pred = clf.predict(X_test)

### Output results
#
print("Outputing results...")
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)


print("Displaying worse results...")
# show top worse results
show_worse = 10
train_ind = list(set(range(0,num_train))-set(rand_ind))
train_pred = clf.predict(X_train[train_ind])
# show over and under estimate
pred_diff = abs(train_pred - y_train[train_ind])
worse_ind = np.argpartition(pred_diff, -1*show_worse)[-1*show_worse:]
# print the worst results to the terminal
print_worse = np.delete(df_train.values[worse_ind],[0,1],1)
print_worse = np.concatenate((print_worse,y_pred[worse_ind].reshape(10,1)),axis=1)
#print_worse = np.concatenate((print_worse,X_train[worse_ind]),axis=1)

print(print_worse)

#
## train a xgboost model too
#param = {'bst:max_depth':6,'bst:eta':0.025}
#num_round = 24
#xg_train = xgboost.DMatrix( X_train, label=y_train)
#clf = xgboost.train( param, xg_train, num_round)
#y_pred = clf.predict(xgboost.DMatrix(X_test))
#
#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('py_xg_sub_.csv',index=False)
#
# Would be nice to see how and what is going wrong

"""
# Should we alter the histogram to be more similar to the initial data?
