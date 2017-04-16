# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:38:11 2016

@author: Beste
"""

import ngram


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
    elif n == 2:
        return sum(int(str2.find(word_ngram)>=0) for word_ngram in ngram.getBiterm(str1.split()," "))
    elif n == 3:
        return sum(int(str2.find(word_ngram)>=0) for word_ngram in ngram.getTriterm(str1.split()," "))
