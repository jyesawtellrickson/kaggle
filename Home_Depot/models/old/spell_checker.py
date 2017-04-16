# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:31:38 2016

@author: Beste
"""

import re, collections
import codecs

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

# need to fix errors with the small and caps

f = codecs.open('dict_attributes.txt',encoding='utf-8')

NWORDS = train(words(f.read()))

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

#def known_edits2(word):
#    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

# Use a simple 1 off correction
def correct(word):
    candidates = known([word]) or known(edits1(word)) or [word]
    return max(candidates, key=NWORDS.get)
    
#def correct(word):
#    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
#    return max(candidates, key=NWORDS.get)