import numpy as np
import matplotlib.pyplot as plt

def plotInitialData (D, classes, histogramMode = True, scatterMode = False, printMode = False):
    '''
    D = data matrix (MxN), classes = classes array.\n
    Plot the initial data for each different parameter and classes.\n
    If just call the function with "D" (initial matrix to plot), it will plot one histogram for each attribute.\n
    If scatterMode == True, it will also print the scatter graph for each combination of parameters.
    '''

    if (histogramMode):

        D0 = D[:, classes==0] # Bad wines
        print(D0)
        D1 = D[:, classes==1] # Good wines

        attrList = ["fixed acidity",
                    "volatile acidity",
                    "citric acid",
                    "residual sugar",
                    "chlorides",
                    "free sulfur dioxide",
                    "total sulfur dioxide",
                    "density",
                    "pH",
                    "sulphates",
                    "alcohol"]
        
        for i in range (len(attrList)):
            plt.figure()
            plt.xlabel(attrList[i])
            plt.hist(D0[i, :], bins = 10, density = True, alpha = 0.4, label = "bad wines")
            plt.hist(D1[i, :], bins = 10, density = True, alpha = 0.4, label = "good wines")
            plt.legend()
            plt.tight_layout()
        plt.show()

    if (scatterMode):
        print("Scatter mode: to be implemented\n")    
    
def plotEstimDensityForRow(attrRow):
    '''
    Uses log-densities (calculated from the function logpdf_GAU_ND for a certain attribute and N samples) to compare with the actual values.\n
    attrRow = (1xN)
    '''

    cov, mu = empirical_cov(attrRow)

    plt.figure()
    plt.hist(attrRow.ravel(), bins=10, density=True)
    XPlot = np.linspace(attrRow.min(), attrRow.max(), 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, cov)))
    plt.show()

def plotEstimDensityAllRows(D):
    '''
    Uses log-densities (calculated from the function logpdf_GAU_ND for all attributes and N samples) to compare with the actual values.\n
    D = (MxN) data matrix.\n
    It will open M plots, one for each attribute comparing with the estimated density X real sample values.
    '''
    attrList = ["fixed acidity",
                "volatile acidity",
                "citric acid",
                "residual sugar",
                "chlorides",
                "free sulfur dioxide",
                "total sulfur dioxide",
                "density",
                "pH",
                "sulphates",
                "alcohol"]

    for i in range (D.shape[0]):
        attr = D[i, :]
        attr = vrow(attr)
        cov, mu = empirical_cov(attr)

        plt.figure()
        plt.xlabel(attrList[i])
        plt.hist(attr.ravel(), bins=10, density=True)
        XPlot = np.linspace(attr.min(), attr.max(), 1000)
        plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, cov)))
    plt.show()

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
        print("Shape of normalized matrix: ",normData.shape)
        # print(np.sum(data, 1))
    
    return normData



def empirical_cov(data, printMode = False):
    '''
        Function to get the empirical covariance from a set of data
    '''
    # Cov function = (1/N)*(x-mu)*(x-mu)^T
    mu = data.mean(1)
    mu = mu.reshape((mu.shape[0], 1))
    Dc = data - mu
    cov = np.dot(Dc,Dc.T)
    cov = cov/float(data.shape[1])
    
    if(printMode):
        print("Covariance max: ", np.max(cov))
        print("Covariance min: ", np.min(cov))
        # print("np.Covariance: ", np.cov(data))
    return cov, mu

def logpdf_GAU_ND(D, mu, C):
    '''
    Calculates the log-densities of data matrix "D".\n
    D = data matrix (MxN == attributes x samples), mu = (Mx1) containing the mean for each attribute, C = (MxM) covariance matrix. \n
    
    '''
    M = D.shape[0]
    T1 = -(M/2)*np.log(2*np.pi)
    T2 = -(1/2)*np.linalg.slogdet(C)[1]
    T3 = -(1/2)*( np.dot(
        np.dot( 
            np.transpose(D-mu), np.linalg.inv(C) ), D-mu) )
    
    T3 = np.diag(T3)
    return T1 + T2 + T3
