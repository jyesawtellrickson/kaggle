import pandas as pd
import numpy as np
from sklearn import linear_model

def seq_to_list(str_seq):
    """
    Convert string sequence to list.

    :param str_seq: sequence string
    :return: sequence list of ints
    """
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


def seq_to_data(seq, n=4):
    """
    Takes a sequence and creates a training/test set.

    Takes a sequence as input and generates many
    training examples of varying length to support
    the model 'learning' the series.

    :param seq: sequence list
    :param n: size of the desired training set int
    :return:
    """
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
    """
    Perform a linear prediction to get the next
    number in the sequence.

    :param seq: a numerical sequence list
    :param print_: flag for enabling printing
    :return: predcition, int
    """
    # get data
    if len(seq) <= 3:
        return 1, seq[-1:][0]
    fil_len = int(len(seq)/2)
    X, y, X_p = seq_to_data(seq, fil_len)

    # Train a linear model
    lmr = linear_model.LinearRegression()
    lmr.fit(X, y)
    # predict next number
    seq_pred = int(round(lmr.predict(X_p)[0],0))
    seq_ans = seq[-1:][0]
    if print_ == 1:
        print('Sequence: ', seq)
        print('My prediction: ', seq_pred)
        print('Correct answer: ', seq_ans)
        print('Predicted accuracy: ', lmr.score(X,y))

    return seq_pred, seq_ans

# Load data
test_df = pd.read_csv('../input/test.csv')
seq_df = pd.read_csv('../input/train.csv')


seq_preds = []
seq_anss = []
# iterate through each sequence and generate a model
for i in range(0, seq_df.shape[0]):
    seq = seq_to_list(seq_df['Sequence'][i])
    seq_pred, seq_ans = lin_pred(seq, 0)
    seq_preds += [seq_pred]
    seq_anss += [seq_ans]
    # let user know progress
    if i % 1000 == 0:
        print("{0:.0f}%".format(100*i/seq_df.shape[0]))

corr = sum([seq_preds[i]==seq_anss[i] for i in range(0,len(seq_preds))])

print("Correct: ", corr, " out of ", len(seq_preds))

"""
def my_score(X,y):
    cor = 0
    for i in range(0,len(y)):
        pred_ = int(round(lmr.predict(X[i])[0],0))
        if pred_ == y[i]:
            cor += 1
"""