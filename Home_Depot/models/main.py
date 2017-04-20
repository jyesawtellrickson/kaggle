import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
# from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
# from nltk.metrics import edit_distance
from nltk.stem.porter import *
# from nltk.stem.snowball import SnowballStemmer
import re
# import enchant
import random
from util import *
# impotr xgboost as xgb

start_time = time.time()
random.seed(2016)
stemmer = PorterStemmer()
# stemmer = SnowballStemmer('english') #0.003 improvement but takes twice as long as PorterStemmer



df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1") #update here
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1") #update here
df_pro_desc = pd.read_csv('../input/product_descriptions.csv') #update here
df_attr = pd.read_csv('../input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
          'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}


RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

#comment out the lines below use df_all.csv for further grid search testing
#if adding features consider any drops on the 'cust_regression_vals' class
#*** would be nice to have a file reuse option or script chaining option on Kaggle Scripts ***

## Data Processing

# apply cleanup and stemming to strings
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stem(x))
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stem(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))

print("--- Stemming: %s minutes ---" % round(((time.time() - start_time)/60), 2))

# combine all product info into one column for processing within map function
df_all['product_info'] = df_all['search_term'] + "\t"+df_all['product_title'] + "\t" +\
                         df_all['product_description']

print("--- Prod Info: %s minutes ---" % round(((time.time() - start_time)/60), 2))

## Feature Generation

# generate phrase length features
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x: len(x.split())).astype(np.int64)
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)

print("--- Len of: %s minutes ---" % round(((time.time() - start_time)/60), 2))

df_all['search_term'] = df_all['product_info'].map(
    lambda x: seg_words(x.split('\t')[0], x.split('\t')[1]))

print("--- Search Term Segment: %s minutes ---" % round(((time.time() - start_time)/60), 2))

# generate features on how often the query is in title or description
df_all['query_in_title'] = df_all['product_info'].map(
    lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[1], 0))
df_all['query_in_description'] = df_all['product_info'].map(
    lambda x: str_whole_word(x.split('\t')[0], x.split('\t')[2], 0))

print("--- Query In: %s minutes ---" % round(((time.time() - start_time)/60), 2))

# generate feature if the last word of the query is in the title or description
df_all['query_last_word_in_title'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0].split(" ")[-1], x.split('\t')[2]))

print("--- Query Last Word In: %s minutes ---" % round(((time.time() - start_time)/60),2))

# generate feature if a query word is int the title or description
df_all['word_in_title'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

# create some ratios to take into account that longer queries are more likely to match
df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']

df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(
    lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
# create feature for brands
df_brand = pd.unique(df_all.brand.ravel())
d = {}
i = 1000
for s in df_brand:
    d[s] = i
    i += 3
df_all['brand_feature'] = df_all['brand'].map(lambda x: d[x])
df_all['search_term_feature'] = df_all['search_term'].map(lambda x: len(x))
# save df_all to avoid reprocessing!
df_all.to_csv('../input/df_all.csv')

## Prediction

#df_all = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)
#num_train = df_all['relevance'].count()
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values
X_train =df_train[:] # shouldn't you remove relevance here? .values
X_test = df_test[:] # .values for other model

print("--- Features Set: %s minutes ---" % round(((time.time() - start_time)/60),2))

y_pred = trainer(X_train, y_train, X_test)

"""
# Old prediction
RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)
rfr = RandomForestRegressor()
model = rfr
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(len(y_pred))
y_pred = np.round(y_pred).astype(int)+1
"""


pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../output/submission.csv', index=False)

print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60), 2))

# from sklearn.feature_extraction import text
# import nltk
"""
df_outliers = pd.read_csv('../input/df_all.csv', encoding="ISO-8859-1", index_col=0)
#stop_ = list(text.ENGLISH_STOP_WORDS)
stop_ = []
d={}
for i in range(len(df_outliers)):
    s = str(df_outliers['search_term'][i]).lower()
    #s = s.replace("\n"," ")
    #s = re.sub("[^a-z]"," ", s)
    #s = s.replace("  "," ")
    a = set(s.split(" "))
    for b_ in a:
        if b_ not in stop_ and len(b_)>0:
            if b_ not in d:
                d[b_] = [1,str_common_word(b_, df_outliers['product_title'][i]),str_common_word(b_, df_outliers['brand'][i]),str_common_word(b_, df_outliers['product_description'][i])]
            else:
                d[b_][0] += 1
                d[b_][1] += str_common_word(b_, df_outliers['product_title'][i])
                d[b_][2] += str_common_word(b_, df_outliers['brand'][i])
                d[b_][3] += str_common_word(b_, df_outliers['product_description'][i])
ds2 = pd.DataFrame.from_dict(d,orient='index')
ds2.columns = ['count','in title','in brand','in prod']
ds2 = ds2.sort_values(['count'], ascending=[False])

f = open("word_review.csv", "w")
f.write("word|count|in title|in brand|in description\n")
for i in range(len(ds2)):
    f.write(ds2.index[i] + "|" + str(ds2["count"][i]) + "|" + str(ds2["in title"][i]) + "|" + str(ds2["in brand"][i]) + "|" + str(ds2["in prod"][i]) + "\n")
f.close()
print("--- Word List Created: %s minutes ---" % round(((time.time() - start_time)/60),2))
"""