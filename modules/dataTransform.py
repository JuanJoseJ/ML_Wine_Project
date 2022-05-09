import numpy as np

def vcol(vlist):
    '''
    Will transform a 1D list into a column vector (rows = len(vlist) // columns = 1)
    1d list = [1,2,3] --vcol--> 
    [[1],
    [2],
    [3]]
    '''
    return np.reshape(vlist, (len(vlist), 1))

def vrow(vlist):
    '''
    Will transform a 1D list into a row vector (rows = 1 // columns = len(vlist))
    1d list = [1,2,3] --vrow--> [[1,2,3]]
    '''
    return np.reshape(vlist, (1, len(vlist)))

def normalize(data):
    '''
    Takes an matrix of attributes MxN and normalizes each tow to fit on [0, 1].
    
    Returns a normalized matrix
    '''
    # normData = []
    # for row in data:
    #     vet = np.divide(row, np.mean)
    #     print(row)
    
    # Normalization formula: norm = (x-min)/(max-min) 
    max_row = np.max(data, 1)
    min_row = np.min(data, 1)
    row_range =max_row-min_row
    dividend = data-vcol(min_row)
    normData = np.divide(dividend, vcol(row_range))
    
    print(normData.shape)
    # print(np.sum(data, 1))
    
    return normData