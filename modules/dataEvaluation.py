from math import sqrt
import numpy as np
from scipy import stats
'''
This file contains all the function related to extracting characteristics from a set of data.
'''

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

# def normal_dist_evaluation(data, printMode = False):
#     '''
#         Calculated the probability that a given set of data follows a normal distribution.
#     '''
#     k2, pval = stats.normaltest(data, 1)
    
#     if(printMode):
#         print("Normal distribution probability: ", pval)
    
#     return pval

def pearson_correlation_coefficient(attrs, labels):
    '''
        Calculates the Pearson Correlation matrix of NxN: The Pearson correlation coefficient is \n
        a number between -1 and 1. In general, the correlation expresses the degree \n
        that, on an average, two variables change correspondingly.
        
        Returns the correlation matrix for both classes
    '''
    c0 = attrs[:, labels==0]
    c1 = attrs[:, labels==1]
    cov0, _ = empirical_cov(c0)
    cov1, _ = empirical_cov(c1)
    
    pcc0 = np.zeros((attrs.shape[0], attrs.shape[0]))
    pcc1 = np.zeros((attrs.shape[0], attrs.shape[0]))
    for i in range(attrs.shape[0]):
        for j in range(attrs.shape[0]):
            corr0_i = cov0[i,j]
            div0 = (np.sqrt(np.var(c0[i,:]))*np.sqrt(np.var(c0[j,:])))
            corr0_i = np.abs(corr0_i/div0)
            pcc0[i,j] = corr0_i
            
            corr1_i = cov1[i,j]
            div1 = (np.sqrt(np.var(c1[i,:]))*np.sqrt(np.var(c1[j,:])))
            corr1_i = np.abs(corr1_i/div1)
            pcc1[i,j] = corr1_i
    
    return pcc0, pcc1