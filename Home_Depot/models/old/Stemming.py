# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:42:27 2016

@author: Beste
"""

# -*- coding: utf-8 -*-

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
import spell_checker

stemmer = SnowballStemmer('english')
#stemmer = LancasterStemmer()

def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

def str_correcter(s):
	return " ".join([spell_checker.correct(word) for word in s.lower().split()])


print("Reading data...")
df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

print("Joining data...")
# Join data together into df_all
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

"""

df_attr_n = df_attr.groupby(['product_uid'])['value','name'].apply(lambda x: ' '.join(x)).reset_index()

"""

# First process attributes to get useful features
df_attr['attributes'] = df_attr['value'].astype(str)
df_attr = df_attr.drop('value',axis=1)
df_attr = df_attr.drop('name',axis=1)
df_attr['attributes'] = df_attr[['product_uid','attributes']].groupby(['product_uid'])['attributes'].transform(lambda x: ' '.join((x)))
df_attr = df_attr[['product_uid','attributes']].drop_duplicates()
df_attr = df_attr.rename(columns = {'attributes':'attributes_values'})

df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')

# First process attributes to get useful features
df_attr = pd.read_csv('../input/attributes.csv')
df_attr['attributes_names'] = df_attr['name'].astype(str)
df_attr = df_attr.drop('value',axis=1)
df_attr = df_attr.drop('name',axis=1)
df_attr['attributes_names'] = df_attr[['product_uid','attributes_names']].groupby(['product_uid'])['attributes_names'].transform(lambda x: ' '.join((x)))
df_attr = df_attr[['product_uid','attributes_names']].drop_duplicates()

df_all = pd.merge(df_all, df_attr, how='left', on='product_uid')


df_all.fillna("Unknown",inplace=True)

# Would it be better to spell check against the search term? Slower, but more accurate. Check difference by one letter.



print("Correcting data...")
# Now go ahead and stem everything
print("   search terms")
df_all['search_term_corrected'] = df_all['search_term'].map(lambda x:str_correcter(x))
#print("   product title")
#df_all['product_title'] = df_all['product_title'].map(lambda x:str_correcter(x))
#print("   product description")
#df_all['product_description'] = df_all['product_description'].map(lambda x:str_correcter(x))
#print("   product attributes values")
#df_all['attributes_values'] = df_all['attributes_values'].astype(str)
#df_all['attributes_values'] = df_all['attributes_values'].map(lambda x:str_correcter(x))
#print("   product attributes names")
#df_all['attributes_names'] = df_all['attributes_names'].astype(str)
#df_all['attributes_names'] = df_all['attributes_names'].map(lambda x:str_correcter(x))



print("Stemming data...")
# Now go ahead and stem everything
print("   search terms")
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
print("   search terms")
df_all['search_term_corrected'] = df_all['search_term_corrected'].map(lambda x:str_stemmer(x))
print("   product title")
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
print("   product description")
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))
print("   product attributes values")
df_all['attributes_values'] = df_all['attributes_values'].astype(str)
df_all['attributes_values'] = df_all['attributes_values'].map(lambda x:str_stemmer(x))
print("   product attributes names")
df_all['attributes_names'] = df_all['attributes_names'].astype(str)
df_all['attributes_names'] = df_all['attributes_names'].map(lambda x:str_stemmer(x))

df_all.info()

df_all.to_csv('../input/df_all_stemmed_corrected.csv')

print("Stemmed data correctly written to csv.")    
    
    
    