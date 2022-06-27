import numpy as np
import scipy.stats

'''
This file contains all the functions that change a set of data, either on shape or content.
'''
def mcol(v):
    return v.reshape((v.size, 1))

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

    print("Finished Gaussianization")    
    return gaussianized
    
def k_folds(data, labels, k=10):
    '''
    ### Explanation
    Returns the k folds of mixed data and labels that can be used for cross validation
    ### Params
    - All the attributes of the data set
    - All the labels of the data set (Corresponding to the attributes)
    - The number of folds to use (Default is 10)
    ### Return
    The tuple of data and labels
    '''
    # Mix the data
    temp_data = np.vstack([data, labels])
    temp_data = np.transpose(temp_data)
    np.random.shuffle(temp_data)
    temp_data = np.transpose(temp_data)
    # Separate data and labels again
    temp_labels = temp_data[-1,:]
    temp_data = np.delete(temp_data, -1, 0)
    # Create the folds
    folds_data = np.split(temp_data, k, axis=1)
    folds_labels = np.split(temp_labels, k)
    return folds_data, folds_labels

def PCA(D, m, verif = False):
    '''
    PCA = Principal Component Analysis. 
    D is the data matrix where columns are the different samples and lines are the attributes of each sample.
    D.shape = MxN
    ##Params
    - D = Data matrix (MxN) where M = number of attributes and N = number of samples
    - m = Number of attributes you want to get
    - verif = Just to print some additional Status
    ## Return
    A matrix of size (mxN)
    '''
    # First step: Calculate mu as the mean of each attribute between all samples.
    mu = D.mean(1)
    mu = vcol(mu)

    # Now it is needed to center the data, i.e., subtract mu (the mean) from all columns of D.
    DC = D - mu

    # Now, it is needed to calculate the covariance matrix C = 1/N * Dc*Dc.T
    N = D.shape[1]
    C = (1/N)*np.dot(DC, np.transpose(DC))

    # Next, we have to compute eigenvectors and eigenvalues:
    sortedEigenValues, eigenVectors = np.linalg.eigh(C) 

    # Note: they are sorted from smallest to largest. We need the opposite:
    # Need to pick the m largest eigenVectors "P" to project the samples into m dimensions:
    P = eigenVectors[:, ::-1][:, 0:m]

    if (verif == True):
        print("Obtained matrix shape:\n", P.shape)

    # Finally, it is needed to apply the projection to a matrix of samples, in this case, "D":
    DP = np.dot(np.transpose(P), D)

    return DP