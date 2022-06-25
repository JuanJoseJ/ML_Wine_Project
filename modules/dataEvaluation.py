from math import sqrt
from unittest import result
import numpy as np
from scipy import stats
import scipy.special as sp
from modules.dataTransform import vrow
import scipy.optimize
'''
This file contains all the function related to extracting characteristics from a set of data and also the models.
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

def log_MVG_Classifier(DT, LT, mu, cov): # Multi Variate Classifier using the logarithms
    S = np.zeros(DT.shape[1])
    for j in range(np.unique(LT).size):
        ll = logpdf_GAU_ND(DT, mu[j], cov[j])
        S = np.vstack((S, ll))
    S = np.delete(S, 0, 0)
    Pc = 1/np.unique(LT).size
    logSJoint = S + np.log(Pc)
    logSMarginal = vrow(sp.logsumexp(logSJoint, axis=0))

    logSPost = logSJoint-logSMarginal # Class posterior probability
    logSPost = np.argmax(logSPost, axis=0);
    return logSPost
#================================= LOGISTIC REGRESSION ==============================================

def logreg_obj(v, DTR, LTR, l, pit):
    '''
    ## Explanation
    This function calculates the function of logistic regression to minimize.
    In this project, considering that the classes are unbalanced, the function receives also pi and nt, nf (number of classes = 1, number of classes = 0)
    ## Params:

    - v = numpy array "(D+1,) = (w,b)" where D = dimensionality of attributes (e.g, D=4 for Iris) and the last column is the b (biases).\n
    - DTR = Training data (M,N).\n
    - LTR = Training labels (N,).\n
    - l = lambda (Multiplier of w).\n
    - pit = Probability of class being 1 (used for unbalanced approaches). If it is 0.5, it is the standard balanced approach.
    '''
    # Retrieving n (number of samples) and nt (number of samples of class 1)
    n = DTR.shape[1]
    nt = np.sum(LTR, 0)
    nf = n - nt

    # Retrieving the weights and biases
    w = v[0:-1]
    b = v[-1]

    temp = l/2*np.linalg.norm(w)**2
    temp2 = 0

    for i in range (n):
        z = 2*LTR[i] - 1 # Which means: z == 1 if class == 1, z == -1 otherwise
        if (LTR[i] == 0):
            temp2 += (1-pit)/nf * np.logaddexp( 0, -z*(np.dot(np.transpose(w),DTR[:,i]) + b) )
        else:
            temp2 += pit/nt * np.logaddexp( 0, -z*(np.dot(np.transpose(w),DTR[:,i]) + b) )


    return temp + temp2

def quadlogreg_obj(v, DTR, LTR, l, pit):
    '''
    ## Explanation
    This function calculates the function of quadratic logistic regression to minimize.
    In this project, considering that the classes are unbalanced, the function receives also pi and nt, nf (number of classes = 1, number of classes = 0)
    ## Params:

    - v = numpy array "(D^2 + D +1,) = (w,b,c)" where D = dimensionality of attributes (e.g, D=4 for Iris) and the second last column is b (size D) and last column is the c (biases).\n
    - DTR = Training data (M,N).\n
    - LTR = Training labels (N,).\n
    - l = lambda (Multiplier of w).\n
    - pit = Probability of class being 1 (used for unbalanced approaches). If it is 0.5, it is the standard balanced approach.
    '''
    # Retrieving n (number of samples) and nt (number of samples of class 1)
    n = DTR.shape[1]
    nt = np.sum(LTR, 0)
    nf = n - nt

    # Retrieving the weights and biases
    w = v[0:-1]
    c = v[-1]

    # Calculating phi(x) = (vec(x*x.T), x)

    temp = l/2*np.linalg.norm(w)**2
    temp2 = 0

    for i in range (n):
        z = 2*LTR[i] - 1 # Which means: z == 1 if class == 1, z == -1 otherwise

        attr = np.reshape(DTR[:,i], (DTR[:,i].shape[0], 1))
        phi = np.dot(attr, attr.T)
        phi = np.hstack(phi.T)
        phi = np.append(phi, attr)

        if (LTR[i] == 0):
            temp2 += (1-pit)/nf * np.logaddexp( 0, -z*(np.dot(np.transpose(w),phi) + c) )
        else:
            temp2 += pit/nt * np.logaddexp( 0, -z*(np.dot(np.transpose(w),phi) + c) )


    return temp + temp2

def posteriorLikelihood(v, DTE, printStats = False, LTE = None, quadratic = False):
    '''
    ## Params:
    - v = numpy array "(D+1,) = (w,b)" where D = dimensionality of attributes (e.g, D=4 for Iris) and the last column is the b (biases). If want to calculate the quadratic form: v = (D^2 + b)\n
    - DTE = Testing dataset (numpy array)\n
    - printStats = Verbose mode for the estimation: Show percentage of correctly assigned classes\n
    - LTE = Only used if printStats == True: LTE is the correct label array\n
    - quadratic = if want to calculate the predicted array for quadratic form of logistic regression

    ## Return:
    Array of predicted labels (0, 0, 1, ...)
    '''
    # Retrieving weight
    w = v[0:-1]

    # Retrieving number of testing samples 
    n = DTE.shape[1]

    if(quadratic):
        c = v[-1]
        predicted = np.zeros((n,))
        for i in range (n):
            attr = np.reshape(DTE[:,i], (DTE[:,i].shape[0], 1))
            phi = np.dot(attr, attr.T)
            phi = np.hstack(phi.T)
            phi = np.append(phi, attr)

            temp = np.dot(np.transpose(w), phi) + c

            if (temp > 0):
                predicted[i] = 1
            else:
                predicted[i] = 0
       
    else:
        b = v[-1]
        predicted = np.dot(np.transpose(w), DTE) + b # Equivalent to likelihood

        for i in range (predicted.shape[0]):
            if (predicted[i] > 0):
                predicted[i] = 1
            else:
                predicted[i] = 0

    if(printStats == True):

        print("======================== LOGISTIC REGRESSION PREDICTION VERBOSE MODE ===============================")
        print("Obtained prediction array:\n")
        print(predicted)
        print("Percentage of correctly assigned classes:\n")
        correct = 0
        for i in range (predicted.shape[0]):
            if (predicted[i] == LTE[i]):
                correct +=1
        print( 100*(correct/LTE.shape[0]), "%" )

    return predicted

def calculateLogReg(DTR, LTR, DTE, LTE, pit, l = 10**(-6), verbose = False, printIterations = False, quadratic = False):
    '''
    ## Explanation
    This function calculates the logistic regression result and returns the array of predicted labels.\n
    First, it uses the scipy.optimize who calls the "logreg_obj" function and minimizes it, returning the estimated position of the minimum (array of weights and bias). OBS: It uses startPoint as being 0.\n
    Finally, it calls "posteriorLikelihood" which performs w.T x DTE and returns the result (predicted labels).

    ## Params
    - DTR = Training data.
    - LTR = Training labels.
    - DTE = Testing data.
    - LTE = Testing labels.
    - pit = Probability of class being 1 (used for unbalanced approaches). If it is 0.5, it is the standard balanced approach.
    - l = lambda (Multiplier of w): by default it is set to 10**(-6).
    - verbose = For debugging: Will print useful information on the terminal. By default is False
    - printIterations = prints the iterations of the algorithm to minimize cost function
    - quadratic = used to set if quadratic logistic regression or not
    '''

    if(quadratic):
        # Retrieving number of attributes D
        D = DTR.shape[0]
        # Generating starting point
        startPoint = np.zeros(D**2 + D + 1)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(quadlogreg_obj, startPoint, iprint = printIterations, args=(DTR, LTR, l, pit), approx_grad=True)

        if (verbose):
            print("Estimated position of the minimum:\n")
            print(x)

        predicted = posteriorLikelihood(x, DTE, verbose, LTE, True)

    else:
        startPoint = np.zeros(DTR.shape[0] + 1)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, startPoint, iprint = printIterations, args=(DTR, LTR, l, pit), approx_grad=True)

        if (verbose):
            print("Estimated position of the minimum:\n")
            print(x)

        predicted = posteriorLikelihood(x, DTE, verbose, LTE)

    return predicted

#====================================================================================================
