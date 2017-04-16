# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
# import xgboost

# Porter stemmer is recommended

# maybe I could take the average of three different predictions to mimic the real setup
# an improvement would use semantic distance
# spelling corrections? synonym replacement
# other replacements, eg. ps 2 -> ps2, hardward related
# maybe divide number of matches by length of descriptions
# try using different ngrams, matching on 2s or 3s of words etc.

# look at difference between predictions and actual, try to fix biggest differences

stemmer = SnowballStemmer('english')

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

num_train = df_train.shape[0]


def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def str_common_word_all(str1, str2):
	return int(str2.find(str(str1))>=0)

# Stem all the words
try:
    df_all = pd.read_csv('../input/df_all_stemmed_corrected.csv')
except OSError:
    print("Stemmed language is missing!")



print("Generating features...")
# Create a length of query which descirbes the number of words in the search term
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
# Create product_info which amalgamates all the information, tab seperated
df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+ \
    df_all['product_description']+"\t"+df_all['attributes_names'].astype(str)+"\t"+ \
    df_all['attributes_values'].astype(str) + "\t" + df_all['search_term_corrected']
# Create feature if the search term is in the product title
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
# As above but with description
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
# As above but with attributes
df_all['word_in_attributes'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[3]))
# As above but with attributes
df_all['word_in_attributes_values'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[4]))

# Create feature if the search term is in the product title
df_all['word_in_title_c'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[5],x.split('\t')[1]))
# As above but with description
df_all['word_in_description_c'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[5],x.split('\t')[2]))
# As above but with attributes
df_all['word_in_attributes_c'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[5],x.split('\t')[3]))
# As above but with attributes
df_all['word_in_attributes_values_c'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[5],x.split('\t')[4]))

# Create feature if the search term is in the product title
df_all['word_in_title'] = df_all[['word_in_title','word_in_title_c']].max(axis=1)
df_all['word_in_description'] = df_all[['word_in_description','word_in_description_c']].max(axis=1)
df_all['word_in_attributes'] = df_all[['word_in_attributes','word_in_attributes_c']].max(axis=1)
df_all['word_in_attributes_values'] = df_all[['word_in_attributes_values','word_in_attributes_values_c']].max(axis=1)

df_all = df_all.drop(['word_in_title_c','word_in_description_c','word_in_attributes_c','word_in_attributes_values_c'],axis=1)

# Should use relative words. i.e. divide by number of words in title/description

# Check if all terms in title
df_all['word_in_title_c'] = df_all['product_info'].map(lambda x:str_common_word_all(x.split('\t')[0],x.split('\t')[1]))
# As above but with description
df_all['word_in_description_c'] = df_all['product_info'].map(lambda x:str_common_word_all(x.split('\t')[0],x.split('\t')[2]))
# As above but with attributes
df_all['word_in_attributes_c'] = df_all['product_info'].map(lambda x:str_common_word_all(x.split('\t')[0],x.split('\t')[3]))
# As above but with attributes
df_all['word_in_attributes_values_c'] = df_all['product_info'].map(lambda x:str_common_word_all(x.split('\t')[0],x.split('\t')[4]))



#df_all = df_all.drop(['Unnamed: 0.1.1','Unnamed: 0.1','Unnamed: 0'],axis=1)
df_all['total_matches'] = (df_all['word_in_title'] + df_all['word_in_description'] + df_all['word_in_attributes'] + df_all['word_in_attributes_values'])*3/4 #+ \
#    df_all['word_in_title_c'] + df_all['word_in_description_c'] + df_all['word_in_attributes_c'] + df_all['word_in_attributes_values_c']
# divide by length
df_all['total_matches'] = df_all['total_matches']*2 / df_all['len_of_query']

print("Preparing for predicting...")
# Remove excess features
df_all = df_all.drop(['search_term','product_title','product_description','product_info','attributes_names','attributes_values','search_term_corrected'],axis=1)
# Remaining: len_of_query, word_in_title, word_in_description


df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']


X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

y_train = df_train['relevance'].values


print("Learning...")
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
print("Predicting...")
y_pred = clf.predict(X_test)

pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

print('Checking results...')

num_cv = num_train

cv_pred = clf.predict(X_train[:num_cv])
cv_est = np.sqrt(sum((cv_pred - y_train[:num_cv])**2)/num_cv)

print("Estimated RMSE = ",cv_est)

show_worse = 10

pred_diff = abs(cv_pred - y_train)
worse_ind = np.argpartition(pred_diff, -1*show_worse)[-1*show_worse:]

df_train.values['worse_ind']

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


# Should we alter the histogram to be more similar to the initial data?
