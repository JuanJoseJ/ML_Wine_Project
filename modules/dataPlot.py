import matplotlib.pyplot as plt
import numpy as np
from .dataEvaluation import logpdf_GAU_ND, empirical_cov
from .dataTransform import vrow

'''
This file contains all the functions that help visualize data by plotting it
'''

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