import numpy as np
from scipy import stats
import scipy.special as sp
from modules.dataTransform import vrow, vcol, mcol
import scipy.optimize
import matplotlib.pyplot as plt
'''
This file contains all the function related to extracting characteristics from a set of data and also the models. 
To execute the main code, please, go to "projectCode.py".
'''

#================================================ DENSITY FUNCTIONS =======================================

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

#===================================================================================================

def empirical_cov(data, printMode = False):
    '''
        Function to get the empirical covariance from a set of data
    '''
    mu = data.mean(1)
    mu = mu.reshape((mu.shape[0], 1))
    Dc = data - mu
    cov = np.dot(Dc,Dc.T)
    cov = cov/float(data.shape[1])
    
    if(printMode):
        print("Covariance max: ", np.max(cov))
        print("Covariance min: ", np.min(cov))
    return cov, mu

def pearson_correlation_coefficient(attrs, labels):
    '''
        Calculates the Pearson Correlation matrix of NxN with binary data:\n
        The Pearson correlation coefficient is a number between -1 and 1. 
        The correlation expresses the degree that, on an average, 
        two variables change correspondingly.\n
        
        Returns the correlation matrix for the raw data and for both classes
    '''
    c0 = attrs[:, labels==0]
    c1 = attrs[:, labels==1]
    cov, _ = empirical_cov(attrs)
    cov0, _ = empirical_cov(c0)
    cov1, _ = empirical_cov(c1)
    
    pcc = np.zeros((attrs.shape[0], attrs.shape[0]))
    pcc0 = np.zeros((attrs.shape[0], attrs.shape[0]))
    pcc1 = np.zeros((attrs.shape[0], attrs.shape[0]))
    for i in range(attrs.shape[0]):
        for j in range(attrs.shape[0]):
            corr_i = cov[i,j]
            div = (np.sqrt(np.var(attrs[i,:]))*np.sqrt(np.var(attrs[j,:])))
            corr_i = np.abs(corr_i/div)
            pcc[i,j] = corr_i

            corr0_i = cov0[i,j]
            div0 = (np.sqrt(np.var(c0[i,:]))*np.sqrt(np.var(c0[j,:])))
            corr0_i = np.abs(corr0_i/div0)
            pcc0[i,j] = corr0_i
            
            corr1_i = cov1[i,j]
            div1 = (np.sqrt(np.var(c1[i,:]))*np.sqrt(np.var(c1[j,:])))
            corr1_i = np.abs(corr1_i/div1)
            pcc1[i,j] = corr1_i
    
    return pcc, pcc0, pcc1

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
        print("Difference between covariances:")
        print(abs(cov0-cov1))
    return result

# ==================================================================================================
def log_MVG_Classifier(DTR, LTR, DTE, prior = [1/2, 1/2]):
    '''
    # Params
    - DTR = Data matrix (training) where columns are the different samples and lines are the attributes of each sample. (M x N)
    - LTR = Label matrix (training) (N,)
    - DTE = Data testing matrix
    - prior = Prior probability for the classes

    # Definition
    Calculates the mean and covariance matrix for each class empirically (used on generative models) and uses it to calculate the log-posterior probabilities for classification.
    In this lab, there are 3 calsses (numbered from 0 to 2).
    '''
    # ============================================================================
    # FIRST STEP: Calcuate the empirical mean and covariance matrix for each class
    # ============================================================================
    
    # Picking samples for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]

    # First step: Calculate mu as the mean of each attribute between all samples.
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))

    # Now it is needed to center the data, i.e., subtract mu (the mean) from all columns of D.
    DC0 = D0 - mu0
    DC1 = D1 - mu1

    # Now, it is needed to calculate the covariance matrix C = 1/N * Dc*Dc.T
    NTR = DTR.shape[1]
    NTE = DTE.shape[1]

    N0 = D0.shape[1]
    C0 = (1/N0)*np.dot(DC0, np.transpose(DC0))

    N1 = D1.shape[1]
    C1 = (1/N1)*np.dot(DC1, np.transpose(DC1))

    # ============================================================================
    # SECOND STEP: Calcuate the log densities for each class and all test samples
    # ============================================================================

    # It will return a score matrix S (C, N) where C = number of classes and N = number of test samples

    S = []
    S.append(logpdf_GAU_ND(DTE, mu0, C0))
    S.append(logpdf_GAU_ND(DTE, mu1, C1))
    S = np.reshape(S, (2, NTE))

    # ============================================================================
    # THIRD STEP: Calcuate the log joint densities (add the score by the prior prob)
    # ============================================================================

    logSJoint = S + np.reshape(np.log(prior), (len(prior), 1))
    
    # ============================================================================
    # FOURTH STEP: Calcuate the posterior probabilities
    # ============================================================================

    logSMarginal =  vrow(scipy.special.logsumexp(logSJoint, axis=0))
    SPost = np.exp(logSJoint - logSMarginal)

    # The predicted labels are obtained picking the highest probability among the classes for a given sample
    predicted = SPost.argmax(0)
    llr = S[1,:] - S[0, :]
    return predicted, llr

def tied_Cov_MVG(DTR, LTR, DTE, prior = [1/2, 1/2]):
    '''
    # Params
    - DTR = Data matrix (training) where columns are the different samples and lines are the attributes of each sample. (M x N)
    - LTR = Label matrix (training) (N,)
    - DTE = Data testing matrix
    - prior = Prior probability for the classes

    # Definition
    Calculates the mean and the tied covariance matrix empirically (used on generative models) and uses it to calculate the log-posterior probabilities for classification.
    In this model, the covariance matrix is the same for all the classes, but the mean of each remains different for each class.
    In this lab, there are 3 calsses (numbered from 0 to 2).
    '''
    # ============================================================================
    # FIRST STEP: Calcuate the empirical mean and tied covariance matrix
    # ============================================================================
    
    # Picking samples for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]

    # First step: Calculate mu as the mean of each attribute between all samples.
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))

    # Now it is needed to center the data, i.e., subtract mu (the mean) from all columns of D.
    DC0 = D0 - mu0
    DC1 = D1 - mu1

    # Now, it is needed to calculate the covariance matrix C = 1/N * Dc*Dc.T
    NTR = DTR.shape[1]
    NTE = DTE.shape[1]

    C0 = np.dot(DC0, np.transpose(DC0))
    C1 = np.dot(DC1, np.transpose(DC1))

    C = (C0+C1)/NTR

    # ============================================================================
    # SECOND STEP: Calcuate the log densities for each class and all test samples
    # ============================================================================

    # It will return a score matrix S (C, N) where C = number of classes and N = number of samples

    S = []
    S.append(logpdf_GAU_ND(DTE, mu0, C))
    S.append(logpdf_GAU_ND(DTE, mu1, C))
    S = np.reshape(S, (2, NTE))

    # ============================================================================
    # THIRD STEP: Calcuate the log joint densities (add the score by the prior prob)
    # ============================================================================

    logSJoint = S + np.reshape(np.log(prior), (len(prior), 1))
    
    # ============================================================================
    # FOURTH STEP: Calcuate the posterior probabilities
    # ============================================================================

    logSMarginal =  vrow(scipy.special.logsumexp(logSJoint, axis=0))
    SPost = np.exp(logSJoint - logSMarginal)

    # The predicted labels are obtained picking the highest probability among the classes for a given sample
    predicted = SPost.argmax(0)
    llr = S[1,:] - S[0, :]

    return predicted, llr

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

    Z = np.where(LTR > 0, 1, -1)
    class1 = np.where(LTR > 0, 1, 0)
    class0 = np.where(LTR > 0, 0, 1)
    temp2 = pit/nt*np.sum(np.logaddexp(0, class1*np.multiply(-Z, np.dot(np.transpose(w),DTR) + b)))
    temp2 += (1-pit)/nf*np.sum(np.logaddexp(0, class0*np.multiply(-Z, np.dot(np.transpose(w),DTR) + b)))
    return temp + temp2


def quadlogreg_obj(v, DTR, LTR, l, pit, PHI):
    '''
    ## Explanation
    This function calculates the function of quadratic logistic regression to minimize.
    In this project, considering that the classes are unbalanced, the function receives also pi and nt, nf (number of classes = 1, number of classes = 0)
    ## Params:

    - v = numpy array "(D^2 + D + 1,) = (w,b,c)" where D = dimensionality of attributes (e.g, D=4 for Iris) and the second last column is b (size D) and last column is the c (biases).\n
    - DTR = Training data (M,N).\n
    - LTR = Training labels (N,).\n
    - l = lambda (Multiplier of w).\n
    - pit = Probability of class being 1 (used for unbalanced approaches). If it is 0.5, it is the standard balanced approach.
    '''
    # Retrieving n (number of samples) and nt (number of samples of class 1)
    n = DTR.shape[1]
    nt = np.sum(LTR, 0)
    nf = n - nt

    # Retrieving number of attributes
    m = DTR.shape[0]

    # Retrieving the weights and biases
    w = v[0:-1]
    c = v[-1]

    # Calculating phi(x) = (vec(x*x.T), x)

    temp = l/2*np.linalg.norm(w)**2
    temp0 = 0
    temp1 = 0

    Z = np.where(LTR > 0, 1, -1)
    class1 = np.where(LTR > 0, 1, 0)
    class0 = np.where(LTR > 0, 0, 1)
    temp2 = pit/nt*np.sum(np.logaddexp(0, class1*np.multiply(-Z, np.dot(np.transpose(w),PHI) + c)))
    temp2 += (1-pit)/nf*np.sum(np.logaddexp(0, class0*np.multiply(-Z, np.dot(np.transpose(w),PHI) + c)))
    # return temp + ((1-pit)/nf)*temp0 + (pit/nt)*temp1
    return temp + temp2

def posteriorLikelihood(v, DTE, printStats = False, LTE = None, quadratic = False, returnScores = False, PHI = None):
    '''
    ## Params:
    - v = numpy array "(D+1,) = (w,b)" where D = dimensionality of attributes (e.g, D=4 for Iris) and the last column is the b (biases). If want to calculate the quadratic form: v = (D^2 + b)\n
    - DTE = Testing dataset (numpy array)\n
    - printStats = Verbose mode for the estimation: Show percentage of correctly assigned classes\n
    - LTE = Only used if printStats == True: LTE is the correct label array\n
    - quadratic = if want to calculate the predicted array for quadratic form of logistic regression
    - returnScores = Set to true if want to return the scores and not the predicted array (0,0,1,1...)

    ## Return:
    Array of predicted labels (0, 0, 1, ...) or scores (1.34, -1.5, ---)
    '''
    # Retrieving weight
    w = v[0:-1]

    # Retrieving number of testing samples 
    n = DTE.shape[1]

    if(quadratic):
        c = v[-1]
        predicted = np.dot(np.transpose(w), PHI) + c # Equivalent to likelihood
        if(returnScores == False):

            for i in range (predicted.shape[0]):
                if (predicted[i] > 0):
                    predicted[i] = 1
                else:
                    predicted[i] = 0
       
    else:
        b = v[-1]
        predicted = np.dot(np.transpose(w), DTE) + b # Equivalent to likelihood
        if(returnScores == False):

            for i in range (predicted.shape[0]):
                if (predicted[i] > 0):
                    predicted[i] = 1
                else:
                    predicted[i] = 0

    if(printStats == True and returnScores == False):

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

def calculatePHI(DTR):
    '''
    
    '''
    
    # Number of samples n
    n = DTR.shape[1]
    
    PHI = np.zeros((DTR.shape[0]**2 + DTR.shape[0], 0))

    for i in range (n):

        attr = np.reshape(DTR[:,i], (DTR[:,i].shape[0], 1))
        phi = np.dot(attr, attr.T)
        phi = np.hstack(phi.T)
        phi = np.append(phi, attr)
        phi = np.reshape(phi, (phi.shape[0], 1))
        PHI = np.hstack((PHI, phi))
    return PHI

def calculateLogReg(DTR, LTR, DTE, LTE, pit, l = 10**(-6), verbose = False, printIterations = False, quadratic = False, returnScores = False, recalibration = False):
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
        PHI = calculatePHI(DTR)

        x, f, d = scipy.optimize.fmin_l_bfgs_b(quadlogreg_obj, startPoint, iprint = printIterations, args=(DTR, LTR, l, pit, PHI), approx_grad=True)

        if (verbose):
            print("Estimated position of the minimum:\n")
            print(x)

        PHI = calculatePHI(DTE)
        predicted = posteriorLikelihood(x, DTE, verbose, LTE, True, returnScores=returnScores, PHI=PHI)

    elif(recalibration == False):
        startPoint = np.zeros(DTR.shape[0] + 1)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, startPoint, iprint = printIterations, args=(DTR, LTR, l, pit), approx_grad=True)

        if (verbose):
            print("Estimated position of the minimum of the expression of linear logistic regression: \n")
            print(x)

        predicted = posteriorLikelihood(x, DTE, verbose, LTE, returnScores=returnScores)
    
    else:
        startPoint = np.zeros(DTR.shape[0] + 1)
        x, f, d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, startPoint, iprint = printIterations, args=(DTR, LTR, l, pit), approx_grad=True)

        if (verbose):
            print("Estimated position of the minimum of the expression of linear logistic regression: \n")
            print(x)

        return x
    return predicted

def plotMinDCFLogReg(DTR, LTR, DTE, LTE, pit, minMaxLambda, resolution, piTilArray = [0.5, 0.1, 0.9], verbose = False, quadratic = False, plot = False):
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
    - minMaxLambda = lambda power (Multiplier of w): list containing min and max of the power of 10 [min, max] = 10**min, 10**max
    - resolution = How many values to consider for the calculated lambdas.
    - piTilArray = Which values to test for different piTil (input for calculation of DCF). By default: [0.5, 0.1, 0.9]
    - verbose = For debugging: Will print useful information on the terminal. By default is False
    - quadratic = used to set if quadratic logistic regression or not
    - plot = If it is true, plot the minDCF right away. If it is false, just returns the array of DCFs (len(piTilArray) x resolution)
    '''

    lambdas = np.logspace(minMaxLambda[0], minMaxLambda[1], resolution)
    print("Lambdas = ", lambdas)
    minDCF = np.zeros((len(piTilArray), resolution))

    maxIt = resolution*len(piTilArray)
    currentIt = 0
    print("number of iterations to take: ", maxIt)


    for i in range (resolution):
        for j in range (len(piTilArray)):
            currentIt += 1
            print("Current iteration = ", currentIt)
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, pit, lambdas[i], verbose, quadratic = quadratic, returnScores = True)
            minDCF[j][i] = bayes_risk(None, piTilArray[j], True, True, predicted, LTE)
            print(minDCF)


    if (plot):
        plt.figure()
        for i in range (len(piTilArray)):
            plt.plot(lambdas, minDCF[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
        plt.xlabel(r"values for $\lambda$")
        plt.ylabel("DCF")
        plt.legend()
        plt.xscale('log')
        plt.show()
        plt.close()
    else:
        return minDCF

def calcLogRegInLambdaRange(DTR, LTR, DTE, LTE, pit, minMaxLambda, resolution, verbose = False, quadratic = False):
    '''
    ## Explanation
    The ideia is to calculate for each lambda the predicted array of labels. This is used to plot the graphs for the minDCF.

    ## Returns
    A matrix "predictions" where number of lines = number of different lambdas considered +1 (the correct labels) and number of columns = number of predicted samples
    '''
    lambdas = np.logspace(minMaxLambda[0], minMaxLambda[1], resolution)
    predictions = np.zeros((resolution, LTE.shape[0]))
   
    maxIt = resolution
    currentIt = 0
    print("number of iterations to take: ", maxIt)


    for i in range (resolution): 
        currentIt += 1
        print("Current iteration = ", currentIt)
        predicted = calculateLogReg(DTR, LTR, DTE, LTE, pit, lambdas[i], verbose, quadratic = quadratic, returnScores = True)
        predictions[i, :] = predicted

    predictions = np.vstack((predictions, LTE))
    return predictions


#====================================================================================================

#============================================= SVM ==================================================
def calcSVMInCRange(DTR, LTR, DTE, LTE, minMaxC, resolution,  K = 1, verbose = False, linear = True, gamma = 1, piT = 0.5, useRebalancing = False, RBF = False):
    '''
    ## Explanation
    The ideia is to calculate for each C the predicted array of labels. This is used to plot the graphs for the minDCF.

    ## Returns
    A matrix "predictions" where number of lines = number of different lambdas considered +1 (the correct labels) and number of columns = number of predicted samples
    '''
    Cs = np.logspace(minMaxC[0], minMaxC[1], resolution)
    predictions = np.zeros((resolution, LTE.shape[0]))
   

    maxIt = resolution
    currentIt = 0
    print("number of iterations to take: ", maxIt)


    for i in range (resolution): 
        currentIt += 1
        print("Current iteration = ", currentIt)
        predicted = calculateSVM(DTR, LTR, Cs[i],DTE, LTE,  K, verbose, linear, gamma, True, piT, useRebalancing, RBF = RBF) # Return scores always true
        predictions[i, :] = predicted

    predictions = np.vstack((predictions, LTE))
    return predictions

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

def kernelSVM_H(DTR, LTR, gamma, K = 0, RBF=False):
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

    if(RBF==False):
        for i in range (n):
            for j in range (n):
                G[i][j] = quadraticKernel(DTR[:, i], DTR[:, j], K**2)
    else:
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

def quadraticKernel(xi, xj, bias, c=1):
    '''
    ## Explanation
    Radial Basis Function kernel: k(x1, x2) = (x1.T*x2 + c)^2 + bias

    ## Params
    - xi and xj = samples (M,) where M is the number of features considered for each sample
    - bias = If want to add a regularized bias on the kernel function
    '''

    quadratic = (np.dot(xi.T, xj) + c)**2 + bias
    return quadratic

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

def  calculateSVM(DTR, LTR, C, DTE, LTE = None, K = 1, verbose = False, linear = True, gamma = 1, returnScores = False, piT = 0.5, useRebalancing = False, RBF=False):
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
    - piT = 
    - useRebalancing = 
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

    if (useRebalancing):
        empPiT = np.sum(LTR)
        empPiT = empPiT/n
        empPiF = 1 - empPiT
        piF = 1 - piT

        for i in range (n):

            z[i][0] = 2*LTR[i] - 1
            param[0][i] = K

            if(LTR[i] == 1):
                bounds.append((0, C*(piT/empPiT)))
            else:
                bounds.append((0, C*(piF/empPiF)))

    else:
        for i in range (n):
            bounds.append((0, C))
            z[i][0] = 2*LTR[i] - 1
            param[0][i] = K

    extD = np.append(DTR, param, 0) # Extended data matrix

    # Retrieving H
    if(linear):
        H = linearSVM_H(DTR, LTR, K)
    elif(RBF):
        H = kernelSVM_H(DTR, LTR, gamma, K, True)
    else:
        H = kernelSVM_H(DTR, LTR, gamma, K, False) # == Quadratic SVM


    # Calculating the minimum
    x,f,d = scipy.optimize.fmin_l_bfgs_b(minDualSVM, startPoint, args = (H, 1), bounds=bounds, iprint = verbose, approx_grad = False, factr=1.0)
    # HAVE TO PASS AT LEAST 2 ARGUMENTS OTHERWISE THIS FUNCTION DOESNT WORK"

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

        if(RBF):
            for i in range(nt):
                for j in range(n):
                    predicted[i] += alpha[j]*z[j][0]*RBFkernel(DTR[:, j],DTE[:, i], gamma, K**2)
        else:
            for i in range(nt):
                for j in range(n):
                    predicted[i] += alpha[j]*z[j][0]*quadraticKernel(DTR[:, j],DTE[:, i], K**2, c=1)
                    
    if (returnScores==False):
        for i in range (predicted.shape[0]):
            if (predicted[i] > 0):
                predicted[i] = 1
            else:
                predicted[i] = 0

    if (verbose and linear and returnScores == False):
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

    elif(verbose and returnScores == False):
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

def confusionMatrix(prediction, correct, printStatus = False, useThreshold = False, threshold = 0):
    '''
    ## Params:
    - prediction = List of predicted classes (assumed to be an 1 dim np array)
    - correct = List of correct classes

    ## Returns:
    Confusion matrix (N,N) where N is the number of classes
    '''
    # Creating empty confusion matrix
    confMatrix = np.zeros((2, 2))

    if (useThreshold):
        prediction = np.where(prediction > threshold, 1, 0)

    # Casting prediction to be integers
    prediction = prediction.astype(int)
    correct = correct.astype(int)

    for i in range (prediction.shape[0]):
        confMatrix[prediction[i]][correct[i]] += 1
    
    if (printStatus):
        print("Percentage of correctly assigned classes:\n")
        correctClasses = 0
        for i in range (prediction.shape[0]):
            if (prediction[i] == correct[i]):
                correctClasses +=1
        print( 100*(correctClasses/correct.shape[0]), "%" )

    return confMatrix

def bayes_risk(confMatrix, piTil, normalized = False, minimum = False, scores = None, labels = None, threshDivision = None, threshold = 0):
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

    
    elif(normalized):

        confMatrix = confusionMatrix(scores, labels, useThreshold=True, threshold = threshold)

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

def logpdf_GMM(X, gmm, returnS = False):
    '''
    ## Overview:
    - The ideia is to approximate each sample of X to a gaussian distribution. In order to do it, this function calls "logpdf_GAU_ND" from lab 4.

    ## Params:
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).

    ## Returns
    If returnS = False, it returns just the log-marginal (which is the sum of S -- See the lab 10). \n
    Else, it returns both S and log marginal
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
        densities += np.log(w)
        S.append(densities)
    
    S = np.reshape(S, (M, N))

    # Now, since everything is log, need to perform a sum of the logs. This means:
    logdens = scipy.special.logsumexp(S, axis=0)

    if (returnS):
        return S, np.reshape(logdens, (1, N))
    else: 
        return np.reshape(logdens, (1, N))

def GMM_EM(X, initialGMM, stop, psi):
    '''
    ## Params
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - initialGMM = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).
    - stop = stop criterion for the algorithm iterations
    - psi = Lower bound to the eigenvalues of the covariance matrices (to make sure that that likelihood doesn't decrease)

    ## Explanation
    EM is an algorithm to calculate an estimation/approximation for a gmm. It uses an approach that is divided into 2 steps: E and M.\n
    - On the E step, it calculates the posterior probability for each component of the given gmm and for each sample. The result is called responsabilities.
    - On the M step, it calculates the new estimated parameters using the responsabilities calculated on the previous step.
    The algorithm runs until the difference of the previous likelihood and the current is lower than the "stop" parameter.
    '''

    # Regtrieving number of samples
    N = X.shape[1]
    D = X.shape[0]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(initialGMM)

    gmm = []

    for i in range(M):
        # Constraining the eigenvalues of the covariance matrices
        w = initialGMM[i][0]
        mu = initialGMM[i][1]
        C = initialGMM[i][2]
        
        U, s, _ = np.linalg.svd(C)
        s[s<psi] = psi
        C = np.dot(U, mcol(s)*U.T)

        gmm.append([w, mu, C])

    # Initial E step:
    S, marginal = logpdf_GMM(X, gmm, returnS=True) # S = (M, N) and marginal = (1, N)
    responsabilities = np.exp(S-marginal) # = Yg,i = (M, N)

    while(True):

        newGMM = []

        prevLikelihood = np.sum(marginal)/N

        # M step:
        for i in range(M):
            Zg = np.sum(responsabilities[i])
            Fg = np.sum(responsabilities[i]*X, 1) # Fg = (D, 1)
            Sg = np.dot(X, (responsabilities[i]*X).T)

            newMu = np.reshape(Fg/Zg, (D, 1))
            newCov = Sg/Zg - np.dot(newMu, newMu.T)
            newW = Zg/(np.sum(responsabilities))

            # Constraining the eigenvalues of the covariance matrices
            U, s, _ = np.linalg.svd(newCov)
            s[s<psi] = psi
            newCov = np.dot(U, mcol(s)*U.T)

            newGMM.append([newW, newMu, newCov])

        # New E step
        S, marginal = logpdf_GMM(X, newGMM, returnS=True) # S = (M, N) and marginal = (1, N)
        newLikelihood = np.sum(marginal)/N

        if(newLikelihood - prevLikelihood < stop):
            break
        else:
            gmm = newGMM
            responsabilities = np.exp(S-marginal)
    
    #print("======================= FINAL LIKELIHOOD = ", newLikelihood)
    return newGMM

def diagonal_GMM_EM(X, initialGMM, stop, psi):
    '''
    ## Params
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).
    - stop = stop criterion for the algorithm iterations

    ## Explanation
    EM is an algorithm to calculate an estimation/approximation for a gmm. It uses an approach that is divided into 2 steps: E and M.\n
    - On the E step, it calculates the posterior probability for each component of the given gmm and for each sample. The result is called responsabilities.
    - On the M step, it calculates the new estimated parameters using the responsabilities calculated on the previous step.
    The algorithm runs until the difference of the previous likelihood and the current is lower than the "stop" parameter.
    '''

    # Regtrieving number of samples
    N = X.shape[1]
    D = X.shape[0]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(initialGMM)

    gmm = []

    for i in range(M):
        # Constraining the eigenvalues of the covariance matrices
        w = initialGMM[i][0]
        mu = initialGMM[i][1]
        C = initialGMM[i][2]
        
        U, s, _ = np.linalg.svd(C)
        s[s<psi] = psi
        C = np.dot(U, mcol(s)*U.T)

        gmm.append([w, mu, C])

    # Initial E step:
    S, marginal = logpdf_GMM(X, gmm, returnS=True) # S = (M, N) and marginal = (1, N)
    responsabilities = np.exp(S-marginal) # = Yg,i = (M, N)

    while(True):

        newGMM = []

        prevLikelihood = np.sum(marginal)/N

        # M step:
        for i in range(M):
            Zg = np.sum(responsabilities[i])
            Fg = np.sum(responsabilities[i]*X, 1) # Fg = (D, 1)
            Sg = np.dot(X, (responsabilities[i]*X).T)

            newMu = np.reshape(Fg/Zg, (D, 1))
            newCov = Sg/Zg - np.dot(newMu, newMu.T)
            newCov = np.multiply(newCov, np.identity(D))
            newW = Zg/(np.sum(responsabilities))

            # Constraining the eigenvalues of the covariance matrices
            U, s, _ = np.linalg.svd(newCov)
            s[s<psi] = psi
            newCov = np.dot(U, mcol(s)*U.T)

            newGMM.append([newW, newMu, newCov])

        # New E step
        S, marginal = logpdf_GMM(X, newGMM, returnS=True) # S = (M, N) and marginal = (1, N)
        newLikelihood = np.sum(marginal)/N

        if(newLikelihood - prevLikelihood < stop):
            break
        else:
            gmm = newGMM
            responsabilities = np.exp(S-marginal)
    
    return newGMM

def tied_GMM_EM(X, initialGMM, stop, psi):
    '''
    ## Params
    - X = (D,N) where D is the size of a sample and N is the number of samples in X;
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]) --> w1 is a value, mu = (D, 1) and C = (D,D).
    - stop = stop criterion for the algorithm iterations

    ## Explanation
    EM is an algorithm to calculate an estimation/approximation for a gmm. It uses an approach that is divided into 2 steps: E and M.\n
    - On the E step, it calculates the posterior probability for each component of the given gmm and for each sample. The result is called responsabilities.
    - On the M step, it calculates the new estimated parameters using the responsabilities calculated on the previous step.
    The algorithm runs until the difference of the previous likelihood and the current is lower than the "stop" parameter.
    '''

    # Regtrieving number of samples
    N = X.shape[1]
    D = X.shape[0]
    # Retrieving M (since gmm is a list and not an array, can't use "shape"):
    M = len(initialGMM)

    gmm = []

    for i in range(M):
        # Constraining the eigenvalues of the covariance matrices
        w = initialGMM[i][0]
        mu = initialGMM[i][1]
        C = initialGMM[i][2]
        
        U, s, _ = np.linalg.svd(C)
        s[s<psi] = psi
        C = np.dot(U, mcol(s)*U.T)

        gmm.append([w, mu, C])

    # Initial E step:
    S, marginal = logpdf_GMM(X, gmm, returnS=True) # S = (M, N) and marginal = (1, N)
    responsabilities = np.exp(S-marginal) # = Yg,i = (M, N)

    while(True):

        newGMM = []

        prevLikelihood = np.sum(marginal)/N

        newWList = []
        newMuList = []
        sumCov = 0
        # M step:
        for i in range(M):
            Zg = np.sum(responsabilities[i])
            Fg = np.sum(responsabilities[i]*X, 1) # Fg = (D, 1)
            Sg = np.dot(X, (responsabilities[i]*X).T)

            newMu = np.reshape(Fg/Zg, (D, 1))
            sumCov += Zg*(Sg/Zg - np.dot(newMu, newMu.T))
            newW = Zg/(np.sum(responsabilities))

            newGMM.append([newW, newMu])

        newCov = sumCov/N
        # Constraining the eigenvalues of the covariance matrices
        U, s, _ = np.linalg.svd(newCov)
        s[s<psi] = psi
        newCov = np.dot(U, mcol(s)*U.T)

        for i in range(M):
            newGMM[i].append(newCov)

        # New E step
        S, marginal = logpdf_GMM(X, newGMM, returnS=True) # S = (M, N) and marginal = (1, N)
        newLikelihood = np.sum(marginal)/N

        if(newLikelihood - prevLikelihood < stop):
            break
        else:
            gmm = newGMM
            responsabilities = np.exp(S-marginal)
    
    return newGMM


def GMM_LBG(X, gmm, alpha, psi, components, stop, type = 'fullCovariance'):
    '''
    ## Explanation
    Given a initial guess for the GMM parameters ([w1, mu1, C1]), it can generate 2, 3, 4... Gs more, so it would become: [[w1, mu1, C1], [w2, mu2, C2], ...]
    With the new generated G, can use it in the EM algorithm to estimate a solution for the model.
    If it receives G, will return 2G.
    ## Params
    - gmm = (M, 3) where M is the number of gaussians that we are considering in the weighted sum (gmm = [[w1, mu1, C1], [w2, mu2, C2], ...]).
    - alpha = factor of multiplication
    - psi = Lower bound to the eigenvalues of the covariance matrices (to make sure that that likelihood doesn't decrease)
    - components = How many components to consider in the prediction
    - stop = stop criterion for the EM algorithm iterations
    '''

    # first optimization using the initial guess
    if (type == 'fullCovariance'):
        newGMM = GMM_EM(X, gmm, stop, psi)
        # print("newGMM = ", newGMM)
    elif(type == 'diagonal'):
        newGMM = diagonal_GMM_EM(X, gmm, stop, psi)
    else:
        newGMM = tied_GMM_EM(X, gmm, stop, psi)

    # newGMM = gmm

    iterations = int(np.log2(components))
    
    for i in range (iterations):
        # Retrieving M (since gmm is a list and not an array, can't use "shape"):
        M = len(newGMM)

        genGMM = []
        for i in range (M):
            w = newGMM[i][0]
            mu = newGMM[i][1]
            C = newGMM[i][2]
            
            # Constraining the eigenvalues of the covariance matrices
            U, s, _ = np.linalg.svd(C)
            s[s<psi] = psi
            C = np.dot(U, mcol(s)*U.T)

            # Calculating displacement vector dg:
            U, s, Vh = np.linalg.svd(C)
            dg = U[:, 0:1] * s[0]**0.5 * alpha

            genGMM.append([w/2, mu + dg, C])
            genGMM.append([w/2, mu - dg, C])

        if (type == 'fullCovariance'):
            newGMM = GMM_EM(X, genGMM, stop, psi)
        elif(type == 'diagonal'):
            newGMM = diagonal_GMM_EM(X, genGMM, stop, psi)
        else:
            newGMM = tied_GMM_EM(X, genGMM, stop, psi)

    return newGMM

def calculateGMM(DTR, DTE, LTR, alpha, psi, components, stop, type = 'fullCovariance'):
    '''
    ## Params
    - DTR
    - DTE
    - initialGMM = [w, mu, C] = Initial gmm to consider
    - alpha = factor of multiplication
    - psi = Lower bound to the eigenvalues of the covariance matrices (to make sure that that likelihood doesn't decrease)
    - components = How many components to consider in the prediction
    - type = 'fullCovariance', 'diagonal' and 'tiedCovariance'
    '''

    # ============================================================================
    # FIRST STEP: Calcuate the empirical mean and covariance matrix for each class
    # ============================================================================
    
    # Picking samples for each class
    D0 = DTR[:, LTR==0]
    D1 = DTR[:, LTR==1]

    # First step: Calculate mu as the mean of each attribute between all samples.
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))

    # Now it is needed to center the data, i.e., subtract mu (the mean) from all columns of D.
    DC0 = D0 - mu0
    DC1 = D1 - mu1

    # Now, it is needed to calculate the covariance matrix C = 1/N * Dc*Dc.T
    NTR = DTR.shape[1]
    NTE = DTE.shape[1]

    N0 = D0.shape[1]
    C0 = (1/N0)*np.dot(DC0, np.transpose(DC0))

    N1 = D1.shape[1]
    C1 = (1/N1)*np.dot(DC1, np.transpose(DC1))

    # ============================================================================
    #        SECOND STEP: Estimate the gmms for the different classes            #
    # ============================================================================

    gmm0 = GMM_LBG(D0, [[1, mu0, C0]], alpha, psi, components, stop, type)
    gmm1 = GMM_LBG(D1, [[1, mu1, C1]], alpha, psi, components, stop, type)

    # ============================================================================
    # THIRD STEP: Calcuate the log densities for each class and all test samples
    # ============================================================================

    # It will return a score matrix S (C, N) where C = number of classes and N = number of samples

    S = []
    S.append(logpdf_GMM(DTE, gmm0))
    S.append(logpdf_GMM(DTE, gmm1))
    S = np.reshape(S, (2, NTE))

    # ============================================================================
    # FOURTH STEP: Calcuate the log joint densities (add the score by the prior prob)
    # ============================================================================

    # logSJoint = S + np.reshape(np.log(prior), (len(prior), 1))
    
    # ============================================================================
    # FOURTH STEP: Calcuate the posterior probabilities
    # ============================================================================

    # logSMarginal =  vrow(scipy.special.logsumexp(logSJoint, axis=0))
    # SPost = np.exp(logSJoint - logSMarginal)

    SPost = np.exp(S)

    # The predicted labels are obtained picking the highest probability among the classes for a given sample
    predicted = SPost.argmax(0)
    llr = S[1,:] - S[0, :]
    
    return predicted, llr
