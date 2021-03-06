from audioop import minmax
from math import sqrt
from unittest import result
import numpy as np
from scipy import stats
import scipy.special as sp
from modules.dataTransform import vrow, vcol, mcol
import scipy.optimize
import matplotlib.pyplot as plt
'''
This file contains all the function related to extracting characteristics from a set of data and also the models.
'''

#================================================ DENSITY FUNCTIONS =======================================

def logpdf_GAU_ND(D, mu, C):
    '''
    Calculates the log-densities of data matrix "D".\n
    D = data matrix (MxN == attributes x samples), mu = (Mx1) containing the mean for each attribute, C = (MxM) covariance matrix. \n
    '''
    #print("Shape of D: ", D.shape, " Shape of mu: ", mu, " Shape of C: ", C)
    M = D.shape[0]
    T1 = -(M/2)*np.log(2*np.pi)
    T2 = -(1/2)*np.linalg.slogdet(C)[1]
    T3 = -(1/2)*( np.dot(
        np.dot( 
            np.transpose(D-mu), np.linalg.inv(C) ), D-mu) )
    
    T3 = np.diag(T3)
    return T1 + T2 + T3

def logpdf_GMM(X, gmm):
    '''
    # Overview:
    - The ideia is to approximate each sample of X to a gaussian distribution. In order to do it, this function calls "logpdf_GAU_ND" from lab 4.
    # Params:
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the wighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]).
    # Returns:
    - S = Matrix of densities
    - logdensities
    '''
    # Retrieving N:
    N = X.shape[1]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(gmm)
    # Creating a matrix S (M,N) to store the log Gau_ND for each mu, C for each sample
    S = []
    for i in range (M):
        w = gmm[i][0]
        mu = gmm[i][1]
        C = gmm[i][2]
        densities = logpdf_GAU_ND(X, mu, C) # Densities is a vector where each element corresponds to log Gau for the current attribute
        densities = densities + np.log(w)
        S.append(densities)
    S = np.reshape(S, (M, N)) # Joint densities matrix
    # Now, since everything is log, need to perform a sum of the logs. This means:
    logdens = scipy.special.logsumexp(S, axis=0) #Log marginal
    return S, np.reshape(logdens, (1, N))

#===================================================================================================

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

def calc_likehoods_ratio(data, mu, cov):
    '''
        Function to calculate the likelihoood ratio to use to calculate the DCF
    '''
    mArr = np.array([]) # Mock array to fill with ll of each class
    ll0 = logpdf_GAU_ND(data, mu[0], cov[0])
    ll1 = logpdf_GAU_ND(data, mu[1], cov[1]) 
    ll = ll1 - ll0
    mArr = np.append(mArr, ll)
    return mArr

def calc_likehoods(data, mu, cov):
    '''
        Function to calculate the likehoood of every sample given some mu and cov
    '''
    mArr = np.array([]) # Mock array to fill with ll of each class
    ll = logpdf_GAU_ND(data, mu, cov)
    mArr = np.append(mArr, ll)
    return mArr
#===================================================================================================
def calc_mu_cov(DTR, LTR):
    '''
    Computes the mu of the training dataset
    '''
    cov = []
    mu = []
    for j in range(np.unique(LTR).size):
        mui = DTR[:,LTR==j].mean(axis=1)
        mui = mui.reshape((mui.shape[0], 1))
        covi, _ = empirical_cov(DTR[:,LTR==j])
        mu.append(mui)
        cov.append(covi)

    return mu, cov

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

# ============================== Full-Cov Gaussean Classifier =================================================

def log_MVG_Classifier(DT, LT, mu, cov):
    '''
        ### Params
        - Data to classify
        - Labels 
        - Training data mean
        - Training data covariance
        ### Return
        The function returns the predicted values for the full-cov gaussean model
    '''
    S = np.zeros(DT.shape[1])
    for j in range(np.unique(LT).size):
        ll = logpdf_GAU_ND(DT, mu[j], cov[j])
        S = np.vstack((S, ll))
    S = np.delete(S, 0, 0)
    Pc = 1/np.unique(LT).size
    logSJoint = S + np.log(Pc)
    logSMarginal = vrow(sp.logsumexp(logSJoint, axis=0))

    logSPost = logSJoint-logSMarginal # Class posterior probability
    prediction = np.argmax(logSPost, axis=0)
    llr = logSPost[1,:] - logSPost[0, :]
    return prediction, llr

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
            print("Estimated position of the minimum of the expression of linear logistic regression: \n")
            print(x)

        predicted = posteriorLikelihood(x, DTE, verbose, LTE)

    return predicted

def plotMinDCFLogReg(DTR, LTR, DTE, LTE, pit, minMaxLambda, resolution, piTilArray = [0.5, 0.1, 0.9], verbose = False, quadratic = False):
    '''
    ## Explanation
    This function calculates the logistic regression result for every lambda between max, min and resolution provided and retrieve its min DCF.\n
    Afterwards, it plots the graph lambda x minDCF
    Finally, it calls "posteriorLikelihood" which performs w.T x DTE and returns the result (predicted labels).

    ## Params
    - DTR = Training data.
    - LTR = Training labels.
    - DTE = Testing data.
    - LTE = Testing labels.
    - pit = Probability of class being 1 (used for unbalanced approaches). If it is 0.5, it is the standard balanced approach.
    - minMaxLambda = lambda (Multiplier of w): list containing min and max to variate [min, max]
    - resolution = How many values to consider for the calculated lambdas.
    - piTilArray = Which values to test for different piTil (input for calculation of DCF). By default: [0.5, 0.1, 0.9]
    - verbose = For debugging: Will print useful information on the terminal. By default is False
    - quadratic = used to set if quadratic logistic regression or not
    '''

    lambdas = np.linspace(minMaxLambda[0], minMaxLambda[1], resolution)
    minDCF = np.zeros((len(piTilArray), resolution))

    maxIt = resolution*len(piTilArray)
    currentIt = 0
    print("number of iterations to take: ", maxIt)


    for i in range (resolution):
        for j in range (len(piTilArray)):
            currentIt += 1
            print("Current iteration = ", currentIt)
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, pit, lambdas[i], verbose, quadratic = quadratic)
            minDCF[j][i] = bayes_risk(None, piTilArray[j], True, True, predicted, LTE)
            print(minDCF)

    plt.figure()
    for i in range (len(piTilArray)):
        plt.plot(lambdas, minDCF[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
    plt.xlabel(r"values for $\lambda$")
    plt.ylabel("DCF")
    plt.legend()
    plt.xscale('log')
    plt.show()
    plt.close()


#====================================================================================================

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
    
    return predicted

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

#=========================== Confusion matrix and detection costs ===================================

def confusionMatrix(prediction, correct):
    '''
    ## Params:
    - prediction = List of predicted classes (assumed to be an 1 dim np array)
    - correct = List of correct classes

    ## Returns:
    Confusion matrix (N,N) where N is the number of classes
    '''
    # Creating empty confusion matrix
    confMatrix = np.zeros((2, 2))

    # Casting prediction to be integers
    prediction = prediction.astype(int)
    correct = correct.astype(int)

    for i in range (prediction.shape[0]):
        # print("AAAAAAAAAAAA", prediction.shape[0])
        # print("BBBBBBBBBBBB", correct.shape[0])
        confMatrix[prediction[i]][correct[i]] += 1

    return confMatrix

def bayes_risk(confMatrix, piTil, normalized = False, minimum = False, scores = None, labels = None, threshDivision = None):
    '''
    ## Quick explanation:
    Evaluation of predictions can be done using empirical Bayes risk (or detection cost function, DCF), that represents the
    cost that we pay due to our decisions c for the test data.

    ## Params:
    - confMatrix = confusion matrix for given attributes on piTil
    - piTil = effective prior (cfn = cfp = 1)
    - normalized = if want to normalize the errors so to get a more accurate value
    - minimum = decides if want to compute the minimum (for this, want also the divisions for threshold variation and the llr)
    - scores = What will be used to set the threshold. If it is a gaussian, use log likilihood ratios. If it is SVM or GMM, use the scores itself. Needed for the calculation of minimum.
    - labels = Testing labels (LTE)
    - threshDivision = how many tests for different threshold.

    ## Returns:
    The result of bayes risk
    '''

    Cfn = 1
    Cfp = 1

    scores = scores.astype(float)
    scoresEval = scores
    scoresEval = np.insert(scores, 0, -np.inf)
    scoresEval = np.insert(scores, len(scores), np.inf)

    if (minimum and normalized): # This is the minimum that the system can achieve given piTil

        DCFValues = []

        for t in range (len(scoresEval)):

            thresh = scoresEval[t]
            predicted = np.where(scores > thresh, 1, 0)
            confMatrix = confusionMatrix(predicted, labels)

            # Retrieve number of false negatives and false positives
            FN = confMatrix[0][1]
            FP = confMatrix[1][0]

            # Retrieving the true positive and true negative values:
            TP = confMatrix[1][1]
            TN = confMatrix[0][0]

            FNR = FN/(FN + TP)
            FPR = FP/(FP + TN)

            DCF = piTil*Cfn*FNR + (1-piTil)*Cfp*FPR
            DCFValues.append(DCF/np.min([piTil*Cfn, (1-piTil)*Cfp]))
        
        return np.min(DCFValues)

    elif(minimum and normalized == False):

        DCFValues = []
        maxThresh = np.max(llr)
        minThresh = np.min(llr)

        for t in range (1, threshDivision, 1):

            thresh = t*( (maxThresh-minThresh)/threshDivision )
            thresh += minThresh
            predicted = np.where(llr > thresh, 1, 0)
            confMatrix = confusionMatrix(predicted, labels)

            # Retrieve number of false negatives and false positives
            FN = confMatrix[0][1]
            FP = confMatrix[1][0]

            # Retrieving the true positive and true negative values:
            TP = confMatrix[1][1]
            TN = confMatrix[0][0]

            FNR = FN/(FN + TP)
            FPR = FP/(FP + TN)

            DCF = piTil*Cfn*FNR + (1-piTil)*Cfp*FPR
            DCFValues.append(DCF)
        
        return np.min(DCFValues)
    
    elif(normalized):

        # Retrieve number of false negatives and false positives
        FN = confMatrix[0][1]
        FP = confMatrix[1][0]

        # Retrieving the true positive and true negative values:
        TP = confMatrix[1][1]
        TN = confMatrix[0][0]

        FNR = FN/(FN + TP)
        FPR = FP/(FP + TN)

        # Doing the actual computation of bayes risk:
        DCF = piTil*Cfn*FNR + (1-piTil)*Cfp*FPR
        return DCF/np.min([piTil*Cfn, (1-piTil)*Cfp])

    elif(not normalized):

        Cfn = 1
        Cfp = 1

        # Retrieve number of false negatives and false positives
        FN = confMatrix[0][1]
        FP = confMatrix[1][0]

        # Retrieving the true positive and true negative values:
        TP = confMatrix[1][1]
        TN = confMatrix[0][0]

        FNR = FN/(FN + TP)
        FPR = FP/(FP + TN)

        # Doing the actual computation of bayes risk:
        DCF = piTil*Cfn*FNR + (1-piTil)*Cfp*FPR
        return DCF

#============================================== GMM ======================================================

def GMM_EM(X, gmm, psi, error=10**-6):
    '''
    ## Explanation
    Calculates a GMM the maximizes the likelihood for the training set.    
    ## Params
    X is a set of attributes and gmm (Initial) is a set of [[w0,mu0,cov0],...]
    ## Returns
    Returns a new GMM with an improved likelihood
    '''
    M = len(gmm)
    S, logdens = logpdf_GMM(X,gmm) # logDens = logMarginal
    logGamma = S-logdens # posteriorLogProb = jointDensities - marginalDensities
    responsability = np.exp(logGamma) # Posterior distribution (MxN). Responsability.
    log_likelihood=None
    newGMM = []
    while log_likelihood is None or abs(np.average(logdens)-np.average(log_likelihood))>error:
        newGMM = []
        for j in range(M):
            gamma = responsability[j,:]
            Zg = np.sum(gamma)
            Fg = np.sum(vrow(gamma)*X, axis=1)
            Sg = np.dot(X, (vrow(gamma)*X).T)
            newMu = vcol(Fg/Zg)
            newCov = (Sg/Zg)-np.dot(newMu,newMu.T)
            newW = Zg/X.shape[1]
            
            # Constraining the eigenvalues of the covariance matrices
            U, s, _ = np.linalg.svd(newCov)
            s[s<psi] = psi
            newCov = np.dot(U, mcol(s)*U.T)
                
            newGMM.append([newW, newMu, newCov])
        log_likelihood = logdens
        S, logdens = logpdf_GMM(X, newGMM)
        logGamma = np.subtract(S, logdens) # 
        responsability = np.exp(logGamma)
        
    return newGMM

def GMM_LBG(gmm, alpha, psi):
    '''
    ## Explanation
    Given a initial guess for the GMM parameters ([w1, mu1, C1]), it can generate 2, 3, 4... Gs more, so it would become: [[w1, mu1, C1], [w2, mu2, C2], ...]
    With the new generated G, can use it in the EM algorithm to estimate a solution for the model.
    If it receives G, will return 2G.
    ## Params
    - gmm = (M, N) where M is the number of gaussians that we are considering in the wighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]).
    - alpha = factor of multiplication
    - psi = Lower bound to the eigenvalues of the covariance matrices (to make sure that that likelihood doesn't decrease)
    '''
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(gmm)
    genGMM = []
    for i in range (M):
        w = gmm[i][0]
        mu = gmm[i][1]
        C = gmm[i][2]
        # Constraining the eigenvalues of the covariance matrices
        # U, s, _ = np.linalg.svd(C)
        # s[s<psi] = psi
        # C = np.dot(U, mcol(s)*U.T)
        # Calculating displacement vector dg:
        U, s, Vh = np.linalg.svd(C)
        dg = U[:, 0:1] * s[0]**0.5 * alpha
        genGMM.append([w/2, mu + dg, C])
        genGMM.append([w/2, mu - dg, C])
    return genGMM

def initial_gmm(DT, LT):
    '''
    ## Explanation:
    This calculates the initial gmm by class for a set of data
    ## Return:
    A matrix of the initial GCC data of shape [[W0,M0,S0],...[Wc,Mc,Sc]]
    '''
    Nc = np.unique(LT).size # Number of classes
    calcGCC = []
    for i in range(Nc):
        DTc = DT[:, LT==i]
        Mc = vcol(DTc.mean(axis=1)) # Class i mean
        Sc = empirical_cov(DTc) # Covariance of class i
        Wc = 1
        calcGCC.append([Wc, Mc, Sc])
    return calcGCC
