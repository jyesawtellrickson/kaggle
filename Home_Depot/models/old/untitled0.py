# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:42:27 2016

@author: Beste
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer
# import xgboost


stemmer = SnowballStemmer('english')


def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())


df_attr = pd.read_csv('../input/attributes.csv')

df_attr['attributes'] = df_attr['value'].astype(str)
df_attr = df_attr.drop('value',axis=1)
df_attr = df_attr.drop('name',axis=1)
df_attr['attributes'] = df_attr[['product_uid','attributes']].groupby(['product_uid'])['attributes'].transform(lambda x: ' '.join((x)))
df_attr = df_attr[['product_uid','attributes']].drop_duplicates()
df_attr = df_attr.rename(columns = {'attributes':'attributes_values'})

print("merging")
# Stem all the words
df_all = pd.read_csv('../input/df_all_stemmed2.csv')
#df_all = df_all.drop('attributes',axis=1)
df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')
df_all.attributes_values.fillna("Unknown",inplace=True)

print("stemming")
df_all['attributes_values'] = df_all['attributes_values'].astype(str)
df_all['attributes_values'] = df_all['attributes_values'].map(lambda x:str_stemmer(x))
df_all.to_csv('../input/df_all_stemmed3.csv')
    
    
    
    
    
    
    



#
#for key, group in groupby(df_attr, lambda x: x[0]):
#    listOfThings = " and ".join([thing[1] for thing in group])
#    print key + "s:  " + listOfThings + "."