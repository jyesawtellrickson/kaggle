import pandas as pd
import numpy as np
from sklearn import linear_model

test_df = pd.read_csv('input/test.csv')
seq_df = pd.read_csv('input/train.csv')

def seq_to_list(str_seq):
    str_seql = str_seq.split(',')
    num_seql = [int(i) for i in str_seql]
    return num_seql

"""
targets_list = seq_df['Sequence'].str.split(',').tolist()
targets = []
for seq in targets_list:
    targets.append(seq[-1:][0])

#print(targets)
#targets_df = seq_df['Id']
#targets_df['target'] = targets
#print(targets_df.head())
"""

# How do we do this?
"""
    Maybe we can run a linear model which predicts the next number based off the previous few.

    This would work for number 2 in an n-gram approach.

    Not all sequences are linear though.

"""

"""
    Takes a sequence a creates a range of values for use in ML problems.
"""
def seq_to_data(seq, n=4):
    X = []
    y = []
    for i in range(0, len(seq)-n-1):
        X.append(seq[i:i+n])
        y.append(seq[i+n])

    X_p = np.array(seq[-(n+1):-1])
    X = np.array(X)
    y = np.array(y)

    return X, y, X_p

def lin_pred(seq, print_= 0):
    # get data
    if len(seq) <= 3:
        return 1, seq[-1:][0]
    fil_len = int(len(seq)/2)
    X, y, X_p = seq_to_data(seq, fil_len)

    # Train a linear model
    lmr = linear_model.LinearRegression()

    lmr.fit(X, y)
    seq_pred = int(round(lmr.predict(X_p)[0],0))
    seq_ans = seq[-1:][0]
    if print_ == 1:
        print('Sequence: ', seq)
        print('My prediction: ', seq_pred)
        print('Correct answer: ', seq_ans)
        print('Predicted accuracy: ', lmr.score(X,y))

    return seq_pred, seq_ans

#seq_pred, seq_ans = lin_pred(seq_2, 0)

seq_preds = []
seq_anss = []
for i in range(0,seq_df.shape[0]):
    seq = seq_to_list(seq_df['Sequence'][i])
    seq_pred, seq_ans = lin_pred(seq, 0)
    seq_preds += [seq_pred]
    seq_anss += [seq_ans]
    if i % 1000 == 0:
        print(i)

#print(seq_preds)
#print(seq_anss)

corr = sum([seq_preds[i]==seq_anss[i] for i in range(0,len(seq_preds))])

print("Correct: ", corr, " out of ", len(seq_preds))

# Can't use basic regression, least squares doesn't accurately hone in. Need to use
# more of a classification style
"""
def my_score(X,y):
    cor = 0
    for i in range(0,len(y)):
        pred_ = int(round(lmr.predict(X[i])[0],0))
        if pred_ == y[i]:
            cor += 1
            """