import numpy as np

'''
This file contains all the functions that change a set of data, either on shape or content.
'''

def vcol(vlist):
    '''
    Will transform a 1D list into a column vector (rows = len(vlist) // columns = 1).\n
    1d list = [1,2,3] --vcol--> \n
    [[1],\n
    [2],\n
    [3]]\n
    '''
    return np.reshape(vlist, (len(vlist), 1))

def vrow(vlist):
    '''
    Will transform a 1D list into a row vector (rows = 1 // columns = len(vlist))
    1d list = [1,2,3] --vrow--> [[1,2,3]]
    '''
    return np.reshape(vlist, (1, len(vlist)))

def normalize(data, printMode = False):
    '''
    Takes an matrix of attributes MxN and normalizes each tow to fit on [0, 1].
    Returns a normalized matrix.
    If want to print the shape of normalized matrix, set printShape = True.
    '''
    # Normalization formula: norm = (x-min)/(max-min) 
    max_row = np.max(data, 1)
    min_row = np.min(data, 1)
    row_range =max_row-min_row
    dividend = data-vcol(min_row)
    normData = np.divide(dividend, vcol(row_range))

    if (printMode):
        print("Normalized matrix: ",normData)
        # print(np.sum(data, 1))
    
    return normData
