import numpy as np
import scipy.stats

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

def gaussianize(attr, printStatus = False):
    '''
    ## Explanation
    It calculates the ranking function of each sample and the percent point function for the training dataset.

    ## Parameters
    - attr = matrix of the attributes (MxN) (ONLY FOR TRAINING)
    - printStatus = if it is true, print the percentage of conclusion

    ## Return
    - Resulting matrix (MxN) of gaussianized features
    '''

    # Retrieving number of attributes and number of samples
    M = attr.shape[0]
    N = attr.shape[1]

    # Initialize the rank and gaussianized arrays
    rank = np.zeros((M, N))
    gaussianized = np.zeros((M, N))

    # Number of times to perform loop (for printing mode only)
    if (printStatus):
        loops = M*N
        loopFrac = loops//100
        loopCount = 0
        percentage = 0
        status = "[----------------------------------------------------------------------------------------------------]"
        addAst = "[****************************************************************************************************]"

    for i in range (M):
        for j in range (N):

            value = attr[i][j]
            sum = 0

            for k in range (N):
                if (attr[i][k] < value):
                    sum += 1
            
            sum += 1
            rank[i][j] = sum/(N+2)
            gaussianized[i][j] = scipy.stats.norm.ppf(rank[i][j])

            if (printStatus):
                loopCount += 1
                if (loopCount%loopFrac == 0):
                    percentage += 1
                    print("A", end = "\r")
                    print ("Progress of gaussianization: ", addAst[0:loopCount//loopFrac + 1] + status[loopCount//loopFrac + 1:], end="\r")

    print("Finished \r")    
    return gaussianized
    

