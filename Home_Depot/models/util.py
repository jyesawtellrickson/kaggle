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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
# from nltk.metrics import edit_distance
from nltk.stem.porter import *
# from nltk.stem.snowball import SnowballStemmer
import re
# import enchant
import random
from util import *

stemmer = PorterStemmer()
# stemmer = SnowballStemmer('english') #0.003 improvement but takes twice as long as PorterStemmer

print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

stop_w = ['for', 'xbi', 'and', 'in', 'th', 'on', 'sku', 'with', 'what', 'from',
          'that', 'less', 'er', 'ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
          'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}


def str_stem(s):
    """
    Perform clean of strings and apply stemming.

    :param s: input string str
    :return: stemmed string str, null if input not str
    """
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        # s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        # convert any numbers to numeric form
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        # apply stemming to each word for matching
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        # convert to lower case
        s = s.lower()
        # correct all spelling mistakes found manually
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")   # can't remember why this one is here
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        return s
    else:
        return "null"


def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]", " ", str2)
    str2 = [z for z in set(str2.split()) if len(z) > 2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word) > 3:
            s1 = []
            s1 += segmentit(word, str2, True)
            if len(s) > 1:
                s += [z for z in s1 if z not in ['er', 'ing', 's', 'less'] and len(z) > 1]
            else:
                s.append(word)
        else:
            s.append(word)
    return " ".join(s)


def segmentit(s, txt_arr, t):
    """

    :param s: a word str
    :param txt_arr: a list of words list of str
    :param t: Boolean
    :return:
    """
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            # check if word is at the start of string
            if word == s[:-j]:
                r.append(s[:-j])
                s = s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    # If non-recursive
    if t:
        i = len(("").join(r))
        if not i == len(st):
            r.append(st[i:])
    return r


def str_common_word(str1, str2):
    """
    Count the number of times a word in str1 appears in str2

    :param str1: search term str
    :param str2: where to search str
    :return: count of words in str1 in str2 int
    """
    words, cnt = str1.split(), 0    # str1.split(" "), 0
    for word in words:
        if str2.find(word) >= 0:
            cnt += 1
    return cnt


def str_whole_word(str1, str2, i_):
    """
    Count the number of times str1 appears in str2

    :param str1: search term str
    :param str2: where to search str
    :param i_: start of search, int
    :return: count of appearances int
    """
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            # match found, continue searching
            cnt += 1
            i_ += len(str1)
    return cnt


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title',
                     'product_description', 'product_info', 'attr', 'brand']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


def trainer(X_train, y_train, X_test):
    '''
    Trainer to predict the product-search relevance.

    :param X_train: training data numpy array
    :param y_train: training solutions numpy array
    :param X_test: test data numpy array
    :return: prediction results numpy array
    '''
    if __name__ == '__main__':
        rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=2016, verbose=1)
        # calculate tf-idf values for n-gram 1
        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        # perform singular value decomposition
        tsvd = TruncatedSVD(n_components=10, random_state=2016)
        # use root mean square error to evaluate
        RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)
        # generate a pipeline to perform transformations
        clf = pipeline.Pipeline([
                ('union', FeatureUnion(
                            transformer_list=[
                                # first get all the calculated features
                                ('cst',  cust_regression_vals()),
                                # generate the tfidf features and perform tsvd
                                ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')),
                                                            ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                                ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')),
                                                            ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                                ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')),
                                                            ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                                ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')),
                                                            ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                                ],
                            transformer_weights={
                                'cst': 1.0,
                                'txt1': 0.5,
                                'txt2': 0.25,
                                'txt3': 0.0,
                                'txt4': 0.5
                                },
                        n_jobs=-1
                        )),
                ('rfr', rfr)])
        # we will now perform grid search to find the best parameters for this model
        param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
        model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid,
                                         n_jobs=-1, cv=2, verbose=20, scoring=RMSE)
        # fit model with best parameters
        model.fit(X_train, y_train)
        # print
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        print(model.best_score_ + 0.47003199274)
        # return the final prediction
        return model.predict(X_test)

