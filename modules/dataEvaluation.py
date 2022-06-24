from math import sqrt
from unittest import result
import numpy as np
from scipy import stats
import scipy.special as sp
from modules.dataTransform import vrow
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

def pearson_correlation_coefficient(attrs, labels):
    '''
        Calculates the Pearson Correlation matrix of NxN with binary data:\n
        The Pearson correlation coefficient is a number between -1 and 1. \n
        In general, the correlation expresses the degree that, on an average, \n
        two variables change correspondingly.\n
        
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

def calc_likehoods(data, mu, cov):
    '''
        Function to calculate the likehoood of every sample given some mu and cov
    '''
    mArr = np.array([]) # Mock array to fill with ll of each class
    ll = logpdf_GAU_ND(data, mu, cov)
    mArr = np.append(mArr, ll)
    return mArr

def comp_cov_matrix(attrs, labels, printMode=False):
    '''
        Compare the covariance matrix for the binary data
    '''
    c0 = attrs[:, labels==0]
    c1 = attrs[:, labels==1]
    cov0, _ = empirical_cov(c0)
    cov1, _ = empirical_cov(c1)
    result = np.array_equal(cov0, cov1)
    
    if(printMode):
        print("Is the covariance for the two classes equal?")
        print(result)
    return result

def log_MVG_Classifier(testData, trainData): # Multi Variate Classifier using the logarithms
    '''
    Parameters: \n
    \t -testData: Both parameters and labels of the evaluation data \n
    \t -trainData: Both parameters and labels of the training data
    Returns: A matrix with the predictions for the evaluation data.
    '''
    # Here I iterate over the different classes of the training data to get the likehoods of each class
    S = np.zeros(testData[0].shape[1]) # Matrix to keep the loglikehoods. Started like this to keep shape
    for j in range(np.unique(testData[1]).size):
        mu = trainData[0][:,trainData[1]==j].mean(1)
        mu = mu.reshape((mu.size, 1))
        cov = empirical_cov(trainData[0][:,trainData[1]==j])
        ll = calc_likehoods(testData[0], mu, cov)
        S = np.vstack((S, ll)) # Contains the ll for each sample in each class
    S = np.delete(S, 0, 0) # pop the zeros
    Pc = 1/3 # Represents the probability of the class being c. It was given and its the same for all classes here.
    # Now we calculate the class posterior probability as logSJoint/SMarginal. This represents the probability
    # that an observation is part of a class given some attributes know a priori.
    logSJoint = S + np.log(Pc) # Creo la matriz de joint densities multiplicando la S por una prior probability DADA
    logSMarginal = vrow(sp.logsumexp(logSJoint, axis=0)) # Its the probability of a sample having its current attriutes

    logSPost = logSJoint-logSMarginal # Class posterior probability
    # Finally we got a matrix with the class probability (row) for each sample (column), 
    # for each column we have to select which row is the one with the highest value. Like this:
    logSPost = np.argmax(logSPost, axis=0); #argmax returns the index of the greatest value in a given axis
    return logSPost