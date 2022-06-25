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
<<<<<<< HEAD

#============================================= SVM ==================================================
def linearSVM_H(DTR, LTR, K = 1):
    '''
    ## Explanation
    Function that calculates the linear SVM. It computes the primal SVM solution through the dual SVM formulation Jb^D.
    Returns H (= zi*zj*xi.T*xj)

    ## Params
    - DTR = Matrix of training data (MxN) where M = number of attributes for each sample and N = number of samples in the training dataset.
    - LTR = Training labels (N,).
    - K = Parameter used to build the extended matrix. Increasing this number may lead to better decisions, but makes the dual problem harder to solve
    '''

    # Retrieving number of samples
    n = DTR.shape[1]

    # Building the extended matrix (adding K at the end of each feature x) and z array
    param = np.zeros((1, n))
    z = np.zeros((n,))
    for i in range(n):
        param[0][i] = K
        z[i] = 2*LTR[i] - 1

    extD = np.append(DTR, param, 0)

    # Computing H: First, will calculate G = xi.T*xj and then multiply by zi*zj
    G  = np.dot(extD.T, extD)
    zizj = z.reshape((n, 1))
    zizj = np.dot(zizj, zizj.T)
    H = np.multiply(zizj, G)

    return H

def kernelSVM_H(DTR, LTR, gamma, K = 0):
    '''
    ## Explanation
    Function that calculates non linear SVM using a kernel function k(xi, xj).
    Returns H (= zi*zj*k(xi, xj))

    ## Params
    - DTR = Matrix of training data (MxN) where M = number of attributes for each sample and N = number of samples in the training dataset.
    - LTR = Training labels (N,).
    - gamma = Hyperparameter used to calculate the kernel function
    - K = If want to add a regularized bias on the kernel function, the bias will be K^2
    '''

    # Retrieving number of samples
    n = DTR.shape[1]

    # Building z array
    z = np.zeros((n,))
    for i in range(n):
        z[i] = 2*LTR[i] - 1

    # Computing H: First, will calculate G = k(xi, xj) and then multiply by zi*zj
    G  = np.zeros((n,n))
    for i in range (n):
        for j in range (n):
            G[i][j] = RBFkernel(DTR[:, i], DTR[:, j], gamma, K**2)

    zizj = z.reshape((n, 1))
    zizj = np.dot(zizj, zizj.T)
    H = np.multiply(zizj, G)

    return H

def RBFkernel(xi, xj, gamma, bias):
    '''
    ## Explanation
    Radial Basis Function kernel: k(x1, x2) = e^( -gamma||xi-xj||^2 ) + b

    ## Params
    - xi and xj = samples (M,) where M is the number of features considered for each sample
    - gamma = Hyperparameter to use on the function
    - bias = If want to add a regularized bias on the kernel function

    '''

    RBF = np.exp( -gamma*(np.linalg.norm(xi - xj)**2) ) + bias
    return RBF

def minDualSVM(alpha, H, test = None):
    '''
    ## Explanation
    Function that calculates -J^D(alpha) = L^D(alpha) (So it can be minimized by scipy) and its gradient.
    Returns a tuple: LD and its gradient.

    ## Params
    - alpha = (N, 1) where N = number of samples
    - H = (N, N) -> Calculated on linearSVM function as zi*zj*xi.T*xj
    - test = variable that does nothing, is just here because otherwise scipy doesnt work and pass a number of arguments = size of H (bad implemented)
    '''

    # Retrieving number of samples
    n = alpha.shape[0]

    alpha = np.reshape(alpha, (n, 1))
    LD = 1/2*np.dot(np.dot(alpha.T, H),alpha) - ( np.dot(alpha.T,np.ones((n,1))) )

    gradient = np.dot(H, alpha) - np.ones((n, 1))
    gradient = np.reshape(gradient, (n,))

    return LD, gradient

def calculateSVM(DTR, LTR, C, DTE, LTE = None, K = 1, verbose = False, linear = True, gamma = 1):
    '''
    ## Explanation
    This functions uses "scipy.optimize.fmin_l_bfgs_b" to minimize L^D. It calls "linearSVM_H" retrieving H which will be used then by function "minDualSVM" to be minimized.
    The resulting alpha will be then used to calculate the primal solution and recover "w*"

    ## Params
    - DTR = Matrix of training data (MxN) where M = number of attributes for each sample and N = number of samples in the training dataset.
    - LTR = Training labels (N,).
    - C = Maximum value for estimated alpha (it is a hyperparameter)
    - DTE = Testing dataset (numpy array)
    - LTE = Only used if printStats == True: LTE is the correct label array
    - K = Parameter used to build the extended matrix. Increasing this number may lead to better decisions, but makes the dual problem harder to solve
    - linear = If set to False, it will use the RBF kernel function to calculate it 
    - gamma = Only used if non linear. It is a hyperparameter used on the calculation of kernel function
    '''
    
    # FIRST PART: Dual solution calculation ------------------------------------------------------------------------------

    # Retrieving number of samples
    n = DTR.shape[1]
    # Retrieving number of testing samples
    nt = DTE.shape[1]


    # Creating bounds for estimated alpha ((min, max), (min, max), ...), calculating z and extended matrix
    bounds = []
    startPoint = np.zeros((n))
    param = np.zeros((1, n))
    z = np.zeros((n,1))

    for i in range (n):
        bounds.append((0, C))
        z[i][0] = 2*LTR[i] - 1
        param[0][i] = K

    extD = np.append(DTR, param, 0) # Extended data matrix

    # Retrieving H
    if(linear):
        H = linearSVM_H(DTR, LTR, K)
    else:
        H = kernelSVM_H(DTR, LTR, gamma, K)


    # Calculating the minimum
    x,f,d = scipy.optimize.fmin_l_bfgs_b(minDualSVM, startPoint, args = (H, 1), bounds=bounds, iprint = verbose, approx_grad = False, factr=1.0)
    # HAVE TO PASS AT LEAST 2 ARGUMENTS OTHERWISE THIS STUPID FUNCTION DOESNT WORK"

    # SECOND PART ------------------------------------------------------------------------------------------------------------
    if(linear):
        # Primal solution calculation 
        alpha = np.reshape(x, (1, n))
        alphaZ = np.multiply(alpha, z.T)

        w = np.dot(alphaZ, extD.T)

        # THIRD PART: Classifying according to w and calculating the predicted classes array ---------------------------------

        param = np.zeros((1, nt))
        for i in range (nt):
            param[0][i] = K

        extT = np.append(DTE, param, 0) # Extended matrix

        # Calculating w.T * xt
        predicted = np.dot(w, extT) # In this case don't need to do do w.T because w is already of shape (1,5)
        predicted = np.reshape(predicted, (nt,))

    else:
        # Calculating for non-linear solution
        predicted = np.zeros((nt,))
        alpha = np.reshape(x, (n,))

        for i in range(nt):
            for j in range(n):
                predicted[i] += alpha[j]*z[j][0]*RBFkernel(DTR[:, j],DTE[:, i], gamma, K**2)

    for i in range (predicted.shape[0]):
        if (predicted[i] > 0):
            predicted[i] = 1
        else:
            predicted[i] = 0

    if (verbose and linear):
        print("======================== LINEAR SVM PREDICTION VERBOSE MODE ===============================")
        print("Obtained prediction of classes:\n")
        print(predicted)
        print("Percentage of correctly assigned classes:\n")
        correct = 0
        for i in range (predicted.shape[0]):
            if (predicted[i] == LTE[i]):
                correct +=1
        print( 100*(correct/LTE.shape[0]), "%" )
        
        gap = dualityGap(w, C, z, extD, f)
        print("Duality gap = ", gap)
        loss, _ = minDualSVM(x, H)
        print("Loss = ", loss)
    elif(verbose):
        print("===================== NON - LINEAR SVM PREDICTION VERBOSE MODE ============================")
        print("Obtained prediction of classes:\n")
        print(predicted)
        print("Percentage of correctly assigned classes:\n")
        correct = 0
        for i in range (predicted.shape[0]):
            if (predicted[i] == LTE[i]):
                correct +=1
        print( 100*(correct/LTE.shape[0]), "%" )
        print("Loss = ", f)

def dualityGap(w, C, z, extD, LD):
    '''
    ## Explanation
    At the optimal solution: J(w) = JD(alpha). So basically this function computes the primal objective j(w) of linear SVM and calculates the gap = j(w) - JD(alpha) (JD == -LD)
    The smaller the duality gap, the more precise is the dual (and thus the primal) solution.

    ## Params
    - w = Calculated weights of linear SVM (1,M) where M = number of attributes of each sample
    - C = Hyperparameter that determines the maximum of alpha (used on "calculateLinaerSVM")
    - z = (N, 1) where N = number of samples and z = -1 if class 0 and z = 1 otherwise
    - extD = Extended matrix of samples where last row is all equal to K (M+1, N)
    - LD = -JD and is retrieved from minDualSVM (is the value of the function at the minimum)
    '''

    # Retrieving number of samples
    n = z.shape[0]

    # Calculating j(w)
    j = 1/2*(np.linalg.norm(w)**2)
    temp = 0
    S = np.dot(w, extD)
    loss = np.maximum(np.zeros(S.shape), 1-np.reshape(z, (n,))*S).sum()
    j = j + C*loss

    gap = j + LD
    return gap
#====================================================================================================

=======
>>>>>>> d46bfc95809c0bcf78bcd2697d2424022956d8b8
