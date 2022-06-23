import numpy as np
from .dataTransform import vcol

'''
This file contains all the functions used to load a set of data and split for training and testing.
'''

def load(address):
    '''
    Load the dataset into a matrix with dimension MxN where M = attributes and N = different samples
    Line retrived from dataset:
    8.1, 0.27, 0.41, 1.45, 0.033, 11,63,0.9908,2.99,0.56,12,0
    8.2, 0.27, 0.41, 1.45, 0.033, 11,63,0.9908,2.99,0.56,12,0
    '''
    dataList = []
    labelList = []
    with open(address) as text:
        for line in text:
            attrs = line.split(',')[0:11]
            attrs = vcol(np.array([float(i) for i in attrs]))
            label = line.split(',')[-1].strip()
            dataList.append(attrs)
            labelList.append(label)
    return np.hstack(dataList), np.array(labelList).astype(int)


def split_db_2to1(D, L, seed=0):
    '''
    ## Explanation
    This functions splits the database in: 2/3 for training and 1/3 for testing.

    ## Params
    - D = Data matrix (MxN) where M = number of features for each sample, N = number of samples.
    - L = Label array (N,).
    - seed = seed used to do the permutation and randomization.
    '''
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)