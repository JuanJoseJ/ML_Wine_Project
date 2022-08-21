from cProfile import label
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from modules.dataTransform import normalize, vrow, vcol, gaussianize, k_folds, PCA
from modules.dataLoad import load, split_db_2to1
from modules.dataEvaluation import logpdf_GAU_ND, empirical_cov, pearson_correlation_coefficient, calculateLogReg, calcLogRegInLambdaRange, calculateSVM, calcSVMInCRange, confusionMatrix, bayes_risk, calc_likehoods_ratio, calc_mu_cov, comp_cov_matrix, log_MVG_Classifier, tied_Cov_MVG, plotMinDCFLogReg, logpdf_GMM, initial_gmm, GMM_EM, GMM_LBG
from modules.dataPlot import plotEstimDensityForRow, plotEstimDensityAllRows, plotInitialData, plotCorrelationHeatMap


def shuffle_data(D,L):
    '''
    ### Description:
    It shuffles the correspondent data and labels to make it more homogenous
    '''
    # Mix the data
    temp_data = np.vstack([D, L])
    temp_data = np.transpose(temp_data)
    np.random.shuffle(temp_data)
    temp_data = np.transpose(temp_data)
    # Separate data and labels again
    L = temp_data[-1,:]
    D = np.delete(temp_data, -1, 0)
    return D, L

def k_fold(D, L, k, choice):
    '''
    # Params
    - D = Data matrix (M, N) where M = number of attributes for each sample and N = number of samples
    - L = Label matrix (N,)
    - k = Number of folds to be considered (integer)
    - choice = what should be done in the kfold
    '''
    D, L = shuffle_data(D, L)

    # Retrieving number of samples:
    N = D.shape[1]
    step = int(N/k)

    if(int(choice)==4): # MVG Classifier
        resultUntied = np.zeros((2,0))
        resultTied = np.zeros((2,0))

        minDCF1List = []
        minDCF5List = []
        minDCF9List = []

        tiedMinDCF1List = []
        tiedMinDCF5List = []
        tiedMinDCF9List = []

        for i in range (k):

            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold #================= Get rid of this part and test
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            # IF gaussianization is requested
            if(choice == 4.1):
                DTE = gaussianize(DTE, DTR)
                DTR = gaussianize(DTR)

            prediction, llr = log_MVG_Classifier(DTR, LTR, DTE)
            temp = np.vstack((llr, LTE))
            resultUntied = np.hstack((resultUntied, temp))

            tiedPrediction, tiedLlr = tied_Cov_MVG(DTR, LTR, DTE)
            temp = np.vstack((tiedLlr, LTE))
            resultTied = np.hstack((resultTied, temp))

        minDCF5 = bayes_risk(None, 0.5, True, True, resultUntied[0,:], resultUntied[1,:], 100)
        minDCF1 = bayes_risk(None, 0.1, True, True, resultUntied[0,:], resultUntied[1,:], 100)
        minDCF9 = bayes_risk(None, 0.9, True, True, resultUntied[0,:], resultUntied[1,:], 100)

        # tiedPrediction, tiedLlr = tied_Cov_MVG(DTR, LTR, DTE)
        tiedMinDCF5 = bayes_risk(None, 0.5, True, True, resultTied[0,:], resultTied[1,:], 100)
        tiedMinDCF1 = bayes_risk(None, 0.1, True, True, resultTied[0,:], resultTied[1,:], 100)
        tiedMinDCF9 = bayes_risk(None, 0.9, True, True, resultTied[0,:], resultTied[1,:], 100)

        # Calculate the risks
        print("Results for MVG:")
        print("Pitil = 0.1: ",minDCF1)
        print("Pitil = 0.5: ",minDCF5)
        print("Pitil = 0.9: ",minDCF9)

        print("Results for tied MVG:")
        print("Pitil = 0.1: ",tiedMinDCF1)
        print("Pitil = 0.5: ",tiedMinDCF5)
        print("Pitil = 0.9: ",tiedMinDCF9)

    # if(choice==5.1): # Plot minDCF without gaussianization for linear logistic regression
    #     resolution = 15
    #     minMaxLambda = [-5, 2]
    #     piTilArray = [0.5, 0.1, 0.9]

    #     minDCFArray5 = np.zeros((k, resolution))
    #     minDCFArray1 = np.zeros((k, resolution))
    #     minDCFArray9 = np.zeros((k, resolution))
    #     finalMinDCFArray = np.zeros((3, resolution))

    #     for i in range (k):
    #         # i will indicate the ith test subset
    #         if (i == k-1): # means that the test set (i) will be the last fold
    #             DTE = D[:, i*step:]
    #             LTE = L[i*step:]
    #             DTR = D[:, 0:i*step]
    #             LTR = L[0:i*step]
    #         else:
    #             DTE = D[:, i*step:(i+1)*step]
    #             LTE = L[i*step:(i+1)*step]
    #             DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
    #             LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

    #         minDCFArray = plotMinDCFLogReg(DTR, LTR, DTE, LTE, 0.5, minMaxLambda, resolution, piTilArray)
    #         minDCFArray5[i, :] = minDCFArray[0, :]
    #         minDCFArray1[i, :] = minDCFArray[1, :]
    #         minDCFArray9[i, :] = minDCFArray[2, :]
        
    #     finalMinDCFArray[0, :] =  minDCFArray5.mean(0)
    #     finalMinDCFArray[1, :] =  minDCFArray1.mean(0)
    #     finalMinDCFArray[2, :] =  minDCFArray9.mean(0)

    #     lambdas = np.logspace(minMaxLambda[0], minMaxLambda[1], resolution)
    #     plt.figure()
    #     for i in range (3): # 3 values of pitil being considered
    #         plt.plot(lambdas, finalMinDCFArray[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
    #     plt.xlabel(r"values for $\lambda$")
    #     plt.ylabel("DCF")
    #     plt.legend()
    #     plt.xscale('log')
    #     plt.show()
    #     plt.close()

    elif(choice==5.1): # Plot minDCF without gaussianization for linear logistic regression
        resolution = 15
        minMaxLambda = [-5, 2]
        lambdas = np.logspace(minMaxLambda[0], minMaxLambda[1], resolution)
        piTilArray = [0.5, 0.1, 0.9]

        predictions = np.zeros((resolution+1, 0))
        finalMinDCFArray = np.zeros((3, resolution))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            predictions = np.hstack((predictions, calcLogRegInLambdaRange(DTR, LTR, DTE, LTE, 0.5, minMaxLambda, resolution)))
        
        for i in range (resolution):
            for j in range (len(piTilArray)):
                finalMinDCFArray[j][i] = bayes_risk(None, piTilArray[j], True, True, predictions[i, :], predictions[-1, :])
            print("calculation ", i, " out of ", resolution)

        plt.figure()
        for i in range (3): # 3 values of pitil being considered
            plt.plot(lambdas, finalMinDCFArray[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
        plt.xlabel(r"values for $\lambda$")
        plt.ylabel("DCF")
        plt.legend()
        plt.xscale('log')
        plt.show()
        plt.close()

    elif(choice==5.2): # Plot minDCF with gaussianization for linear logistic regression
        resolution = 15
        minMaxLambda = [-5, 2]
        lambdas = np.logspace(minMaxLambda[0], minMaxLambda[1], resolution)
        piTilArray = [0.5, 0.1, 0.9]

        predictions = np.zeros((resolution+1, 0))
        finalMinDCFArray = np.zeros((3, resolution))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            DTE = gaussianize(DTE, DTR)
            DTR = gaussianize(DTR)

            predictions = np.hstack((predictions, calcLogRegInLambdaRange(DTR, LTR, DTE, LTE, 0.5, minMaxLambda, resolution)))
            print("Fold number ", i, " done...")
        
        for i in range (resolution):
            for j in range (len(piTilArray)):
                finalMinDCFArray[j][i] = bayes_risk(None, piTilArray[j], True, True, predictions[i, :], predictions[-1, :])
            
            print("calculation ", i, " out of ", resolution)

        plt.figure()
        for i in range (3): # 3 values of pitil being considered
            plt.plot(lambdas, finalMinDCFArray[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
        plt.xlabel(r"values for $\lambda$")
        plt.ylabel("DCF")
        plt.legend()
        plt.xscale('log')
        plt.show()
        plt.close()

    elif(choice==5.3): # log reg without gaussianization
        l = 10**(-5)
        finalMinDCFArray = np.zeros((3, 3))

        dcf_list5LLogReg = np.zeros((3, 1))
        dcf_list1LLogReg = np.zeros((3, 1))
        dcf_list9LLogReg = np.zeros((3, 1))

        resultsForPit5 = np.zeros((2, 0))
        resultsForPit1 = np.zeros((2, 0))
        resultsForPit9 = np.zeros((2, 0))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.5, l, False, False, False, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit5 = np.hstack((resultsForPit5, temp))
            # dcf_list5LLogReg[0][i] = bayes_risk(None, 0.5, True, True, predicted, LTE)
            # dcf_list1LLogReg[0][i] = bayes_risk(None, 0.1, True, True, predicted, LTE)
            # dcf_list9LLogReg[0][i] = bayes_risk(None, 0.9, True, True, predicted, LTE)

            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.1, l, False, False, False, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit1 = np.hstack((resultsForPit1, temp))
            # dcf_list5LLogReg[1][i] = bayes_risk(None, 0.5, True, True, predicted, LTE)
            # dcf_list1LLogReg[1][i] = bayes_risk(None, 0.1, True, True, predicted, LTE)
            # dcf_list9LLogReg[1][i] = bayes_risk(None, 0.9, True, True, predicted, LTE)

            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.9, l, False, False, False, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit9 = np.hstack((resultsForPit9, temp))
            # dcf_list5LLogReg[2][i] = bayes_risk(None, 0.5, True, True, predicted, LTE)
            # dcf_list1LLogReg[2][i] = bayes_risk(None, 0.1, True, True, predicted, LTE)
            # dcf_list9LLogReg[2][i] = bayes_risk(None, 0.9, True, True, predicted, LTE)
        
            print("Fold number ", i)

        # dcf_list5LLogReg = np.reshape(dcf_list5LLogReg.mean(1), (3,1)) # For piTil = 0.5
        # dcf_list1LLogReg = np.reshape(dcf_list1LLogReg.mean(1), (3,1)) # For piTil = 0.1
        # dcf_list9LLogReg = np.reshape(dcf_list9LLogReg.mean(1), (3,1)) # For piTil = 0.9

        dcf_list5LLogReg[0][0] = bayes_risk(None, 0.5, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list5LLogReg[1][0] = bayes_risk(None, 0.5, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list5LLogReg[2][0] = bayes_risk(None, 0.5, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list1LLogReg[0][0] = bayes_risk(None, 0.1, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list1LLogReg[1][0] = bayes_risk(None, 0.1, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list1LLogReg[2][0] = bayes_risk(None, 0.1, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list9LLogReg[0][0] = bayes_risk(None, 0.9, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list9LLogReg[1][0] = bayes_risk(None, 0.9, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list9LLogReg[2][0] = bayes_risk(None, 0.9, True, True, resultsForPit9[0, :], resultsForPit9[1, :])
        
        finalMinDCFArray = np.hstack((dcf_list5LLogReg, dcf_list1LLogReg))
        finalMinDCFArray = np.hstack((finalMinDCFArray, dcf_list9LLogReg))
        print("final DCFs for linear logistic regression:")
        print("For pit = 0.5:", finalMinDCFArray[0])
        print("For pit = 0.1:", finalMinDCFArray[1])
        print("For pit = 0.9:", finalMinDCFArray[2])

    elif(choice==5.4): # log reg with gaussianization
        l = 10**(-5)

        finalMinDCFArray = np.zeros((3, 3))

        dcf_list5LLogReg = np.zeros((3, 1))
        dcf_list1LLogReg = np.zeros((3, 1))
        dcf_list9LLogReg = np.zeros((3, 1))

        resultsForPit5 = np.zeros((2, 0))
        resultsForPit1 = np.zeros((2, 0))
        resultsForPit9 = np.zeros((2, 0))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            DTE = gaussianize(DTE, DTR)
            DTR = gaussianize(DTR)

            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.5, l, False, False, False, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit5 = np.hstack((resultsForPit5, temp))
            
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.1, l, False, False, False, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit1 = np.hstack((resultsForPit1, temp))
            
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.9, l, False, False, False, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit9 = np.hstack((resultsForPit9, temp))
            
            print("Fold number ", i)

        dcf_list5LLogReg[0][0] = bayes_risk(None, 0.5, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list5LLogReg[1][0] = bayes_risk(None, 0.5, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list5LLogReg[2][0] = bayes_risk(None, 0.5, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list1LLogReg[0][0] = bayes_risk(None, 0.1, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list1LLogReg[1][0] = bayes_risk(None, 0.1, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list1LLogReg[2][0] = bayes_risk(None, 0.1, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list9LLogReg[0][0] = bayes_risk(None, 0.9, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list9LLogReg[1][0] = bayes_risk(None, 0.9, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list9LLogReg[2][0] = bayes_risk(None, 0.9, True, True, resultsForPit9[0, :], resultsForPit9[1, :])
        
        finalMinDCFArray = np.hstack((dcf_list5LLogReg, dcf_list1LLogReg))
        finalMinDCFArray = np.hstack((finalMinDCFArray, dcf_list9LLogReg))
        print("final DCFs for linear logistic regression:")
        print("For pit = 0.5:", finalMinDCFArray[0])
        print("For pit = 0.1:", finalMinDCFArray[1])
        print("For pit = 0.9:", finalMinDCFArray[2])

    elif(choice==6.1): # Plot minDCF without gaussianization for quad logistic regression
        resolution = 15
        minMaxLambda = [-5, 2]
        lambdas = np.logspace(minMaxLambda[0], minMaxLambda[1], resolution)
        piTilArray = [0.5, 0.1, 0.9]

        predictions = np.zeros((resolution+1, 0))
        finalMinDCFArray = np.zeros((3, resolution))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            predictions = np.hstack((predictions, calcLogRegInLambdaRange(DTR, LTR, DTE, LTE, 0.5, minMaxLambda, resolution, quadratic=True)))
            print("Fold number ", i, " done...")
        
        for i in range (resolution):
            for j in range (len(piTilArray)):
                finalMinDCFArray[j][i] = bayes_risk(None, piTilArray[j], True, True, predictions[i, :], predictions[-1, :])
            
            print("calculation ", i, " out of ", resolution)

        plt.figure()
        for i in range (3): # 3 values of pitil being considered
            plt.plot(lambdas, finalMinDCFArray[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
        plt.xlabel(r"values for $\lambda$")
        plt.ylabel("DCF")
        plt.legend()
        plt.xscale('log')
        plt.show()
        plt.close()

    elif(choice==6.2): # Plot minDCF with gaussianization for quad logistic regression
        resolution = 15
        minMaxLambda = [-5, 2]
        lambdas = np.logspace(minMaxLambda[0], minMaxLambda[1], resolution)
        piTilArray = [0.5, 0.1, 0.9]

        predictions = np.zeros((resolution+1, 0))
        finalMinDCFArray = np.zeros((3, resolution))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            DTE = gaussianize(DTE, DTR)
            DTR = gaussianize(DTR)

            predictions = np.hstack((predictions, calcLogRegInLambdaRange(DTR, LTR, DTE, LTE, 0.5, minMaxLambda, resolution, quadratic=True)))
            print("Fold number ", i, " done...")
        
        for i in range (resolution):
            for j in range (len(piTilArray)):
                finalMinDCFArray[j][i] = bayes_risk(None, piTilArray[j], True, True, predictions[i, :], predictions[-1, :])
            
            print("calculation ", i, " out of ", resolution)

        plt.figure()
        for i in range (3): # 3 values of pitil being considered
            plt.plot(lambdas, finalMinDCFArray[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
        plt.xlabel(r"values for $\lambda$")
        plt.ylabel("DCF")
        plt.legend()
        plt.xscale('log')
        plt.show()
        plt.close()

    elif(choice==6.3): # quad log reg without gaussianization
        l = 10**(-3)

        finalMinDCFArray = np.zeros((3, 3))

        dcf_list5LLogReg = np.zeros((3, 1))
        dcf_list1LLogReg = np.zeros((3, 1))
        dcf_list9LLogReg = np.zeros((3, 1))

        resultsForPit5 = np.zeros((2, 0))
        resultsForPit1 = np.zeros((2, 0))
        resultsForPit9 = np.zeros((2, 0))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.5, l, False, False, True, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit5 = np.hstack((resultsForPit5, temp))
            
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.1, l, False, False, True, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit1 = np.hstack((resultsForPit1, temp))
            
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.9, l, False, False, True, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit9 = np.hstack((resultsForPit9, temp))
            
            print("Fold number ", i)

        dcf_list5LLogReg[0][0] = bayes_risk(None, 0.5, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list5LLogReg[1][0] = bayes_risk(None, 0.5, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list5LLogReg[2][0] = bayes_risk(None, 0.5, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list1LLogReg[0][0] = bayes_risk(None, 0.1, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list1LLogReg[1][0] = bayes_risk(None, 0.1, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list1LLogReg[2][0] = bayes_risk(None, 0.1, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list9LLogReg[0][0] = bayes_risk(None, 0.9, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list9LLogReg[1][0] = bayes_risk(None, 0.9, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list9LLogReg[2][0] = bayes_risk(None, 0.9, True, True, resultsForPit9[0, :], resultsForPit9[1, :])
        
        finalMinDCFArray = np.hstack((dcf_list5LLogReg, dcf_list1LLogReg))
        finalMinDCFArray = np.hstack((finalMinDCFArray, dcf_list9LLogReg))
        print("final DCFs for quadratic logistic regression:")
        print("For pit = 0.5:", finalMinDCFArray[0])
        print("For pit = 0.1:", finalMinDCFArray[1])
        print("For pit = 0.9:", finalMinDCFArray[2])
    
    elif(choice==6.4): # quad log reg with gaussianization
        l = 10**(-3)

        finalMinDCFArray = np.zeros((3, 3))

        dcf_list5LLogReg = np.zeros((3, 1))
        dcf_list1LLogReg = np.zeros((3, 1))
        dcf_list9LLogReg = np.zeros((3, 1))

        resultsForPit5 = np.zeros((2, 0))
        resultsForPit1 = np.zeros((2, 0))
        resultsForPit9 = np.zeros((2, 0))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            DTE = gaussianize(DTE, DTR)
            DTR = gaussianize(DTR)

            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.5, l, False, False, True, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit5 = np.hstack((resultsForPit5, temp))
            
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.1, l, False, False, True, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit1 = np.hstack((resultsForPit1, temp))
            
            predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.9, l, False, False, True, True)
            temp = np.vstack((predicted, LTE))
            resultsForPit9 = np.hstack((resultsForPit9, temp))
            
            print("Fold number ", i)

        dcf_list5LLogReg[0][0] = bayes_risk(None, 0.5, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list5LLogReg[1][0] = bayes_risk(None, 0.5, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list5LLogReg[2][0] = bayes_risk(None, 0.5, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list1LLogReg[0][0] = bayes_risk(None, 0.1, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list1LLogReg[1][0] = bayes_risk(None, 0.1, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list1LLogReg[2][0] = bayes_risk(None, 0.1, True, True, resultsForPit9[0, :], resultsForPit9[1, :])

        dcf_list9LLogReg[0][0] = bayes_risk(None, 0.9, True, True, resultsForPit5[0, :], resultsForPit5[1, :])
        dcf_list9LLogReg[1][0] = bayes_risk(None, 0.9, True, True, resultsForPit1[0, :], resultsForPit1[1, :])
        dcf_list9LLogReg[2][0] = bayes_risk(None, 0.9, True, True, resultsForPit9[0, :], resultsForPit9[1, :])
        
        finalMinDCFArray = np.hstack((dcf_list5LLogReg, dcf_list1LLogReg))
        finalMinDCFArray = np.hstack((finalMinDCFArray, dcf_list9LLogReg))
        print("final DCFs for quadratic logistic regression:")
        print("For pit = 0.5:", finalMinDCFArray[0])
        print("For pit = 0.1:", finalMinDCFArray[1])
        print("For pit = 0.9:", finalMinDCFArray[2])

    elif(int(choice)==7): # Plots for the linear SVM
        resolution = 15
        minMaxC = [-3, 1]
        Cs = np.logspace(minMaxC[0], minMaxC[1], resolution)
        piTilArray = [0.5, 0.1, 0.9]

        predictions = np.zeros((resolution+1, 0))
        finalMinDCFArray = np.zeros((3, resolution))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            if(choice==7.1):
                predictions = np.hstack((predictions, calcSVMInCRange(DTR, LTR, DTE, LTE, minMaxC, resolution)))

            elif(choice==7.2):
                predictions = np.hstack((predictions, calcSVMInCRange(DTR, LTR, DTE, LTE, minMaxC, resolution, useRebalancing=True)))

            elif(choice==7.3):
                DTE = gaussianize(DTE, DTR)
                DTR = gaussianize(DTR)
                predictions = np.hstack((predictions, calcSVMInCRange(DTR, LTR, DTE, LTE, minMaxC, resolution)))

            elif(choice==7.4):
                DTE = gaussianize(DTE, DTR)
                DTR = gaussianize(DTR)
                predictions = np.hstack((predictions, calcSVMInCRange(DTR, LTR, DTE, LTE, minMaxC, resolution, useRebalancing=True)))

            
            print("Fold number ", i, " done...")
        
        for i in range (resolution):
            for j in range (len(piTilArray)):
                finalMinDCFArray[j][i] = bayes_risk(None, piTilArray[j], True, True, predictions[i, :], predictions[-1, :])
            
            print("calculation ", i, " out of ", resolution)

        plt.figure()
        for i in range (3): # 3 values of pitil being considered
            plt.plot(Cs, finalMinDCFArray[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
        plt.xlabel(r"values for C$")
        plt.ylabel("DCF")
        plt.legend()
        plt.xscale('log')
        plt.show()
        plt.close()

    elif(choice==8): # linear SVM with gaussianization
        C = 10**(0)

        finalMinDCFArray = np.zeros((2, 3))

        dcf_list5 = np.zeros((2, 1))
        dcf_list1 = np.zeros((2, 1))
        dcf_list9 = np.zeros((2, 1))

        resultsForUnbalanced = np.zeros((2, 0))
        resultsForBalanced = np.zeros((2, 0))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            DTE = gaussianize(DTE, DTR)
            DTR = gaussianize(DTR)

            # For unbalanced SVM
            predicted = calculateSVM(DTR, LTR, C, DTE, LTE, returnScores=True)
            temp = np.vstack((predicted, LTE))
            resultsForUnbalanced = np.hstack((resultsForUnbalanced, temp))

            # For balanced SVM
            predicted = calculateSVM(DTR, LTR, C, DTE, LTE, returnScores=True, useRebalancing=True)
            temp = np.vstack((predicted, LTE))
            resultsForBalanced = np.hstack((resultsForBalanced, temp))
            
            
            print("Fold number ", i)

        dcf_list5[0][0] = bayes_risk(None, 0.5, True, True, resultsForUnbalanced[0, :], resultsForUnbalanced[1, :])
        dcf_list5[1][0] = bayes_risk(None, 0.5, True, True, resultsForBalanced[0, :], resultsForBalanced[1, :])

        dcf_list1[0][0] = bayes_risk(None, 0.1, True, True, resultsForUnbalanced[0, :], resultsForUnbalanced[1, :])
        dcf_list1[1][0] = bayes_risk(None, 0.1, True, True, resultsForBalanced[0, :], resultsForBalanced[1, :])

        dcf_list9[0][0] = bayes_risk(None, 0.9, True, True, resultsForUnbalanced[0, :], resultsForUnbalanced[1, :])
        dcf_list9[1][0] = bayes_risk(None, 0.9, True, True, resultsForBalanced[0, :], resultsForBalanced[1, :])
        
        finalMinDCFArray = np.hstack((dcf_list5, dcf_list1))
        finalMinDCFArray = np.hstack((finalMinDCFArray, dcf_list9))
        print("final DCFs for quadratic logistic regression:")
        print("For Unbalanced:", finalMinDCFArray[0])
        print("For Balanced:", finalMinDCFArray[1])
    
    elif(int(choice)==9): # Plots for the quadratic SVM
        resolution = 15
        minMaxC = [-3, 1]
        Cs = np.logspace(minMaxC[0], minMaxC[1], resolution)
        piTilArray = [0.5, 0.1, 0.9]

        predictions = np.zeros((resolution+1, 0))
        finalMinDCFArray = np.zeros((3, resolution))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            DTE = gaussianize(DTE, DTR)
            DTR = gaussianize(DTR)

            if(choice==9.1): # Without rebalancing
                predictions = np.hstack((predictions, calcSVMInCRange(DTR, LTR, DTE, LTE, minMaxC, resolution, linear=False, RBF=False)))

            elif(choice==9.2): # With rebalancing
                predictions = np.hstack((predictions, calcSVMInCRange(DTR, LTR, DTE, LTE, minMaxC, resolution, useRebalancing=True, linear=False, RBF=False)))

            
            print("Fold number ", i, " done...")
        
        for i in range (resolution):
            for j in range (len(piTilArray)):
                finalMinDCFArray[j][i] = bayes_risk(None, piTilArray[j], True, True, predictions[i, :], predictions[-1, :])
            
            print("calculation ", i, " out of ", resolution)

        plt.figure()
        for i in range (3): # 3 values of pitil being considered
            plt.plot(Cs, finalMinDCFArray[i, :], label=r"minDCF ($\tilde \pi$ = %f)" %piTilArray[i])
        plt.xlabel(r"values for C$")
        plt.ylabel("DCF")
        plt.legend()
        plt.xscale('log')
        plt.show()
        plt.close()

    elif(choice==10): # quad SVM with gaussianization
        C = 10**(0)

        finalMinDCFArray = np.zeros((2, 3))

        dcf_list5 = np.zeros((2, 1))
        dcf_list1 = np.zeros((2, 1))
        dcf_list9 = np.zeros((2, 1))

        resultsForUnbalanced = np.zeros((2, 0))
        resultsForBalanced = np.zeros((2, 0))

        for i in range (k):
            # i will indicate the ith test subset
            if (i == k-1): # means that the test set (i) will be the last fold
                DTE = D[:, i*step:]
                LTE = L[i*step:]
                DTR = D[:, 0:i*step]
                LTR = L[0:i*step]
            else:
                DTE = D[:, i*step:(i+1)*step]
                LTE = L[i*step:(i+1)*step]
                DTR = np.hstack( (D[:, 0:i*step], D[:, (i+1)*step:]) )
                LTR = np.hstack( (L[0:i*step], L[(i+1)*step:]) )

            DTE = gaussianize(DTE, DTR)
            DTR = gaussianize(DTR)

            # For unbalanced SVM
            predicted = calculateSVM(DTR, LTR, C, DTE, LTE, returnScores=True, RBF=False, linear=False)
            temp = np.vstack((predicted, LTE))
            resultsForUnbalanced = np.hstack((resultsForUnbalanced, temp))

            # For balanced SVM
            predicted = calculateSVM(DTR, LTR, C, DTE, LTE, returnScores=True, useRebalancing=True, RBF=False, linear=False)
            temp = np.vstack((predicted, LTE))
            resultsForBalanced = np.hstack((resultsForBalanced, temp))
            
            
            print("Fold number ", i)

        dcf_list5[0][0] = bayes_risk(None, 0.5, True, True, resultsForUnbalanced[0, :], resultsForUnbalanced[1, :])
        dcf_list5[1][0] = bayes_risk(None, 0.5, True, True, resultsForBalanced[0, :], resultsForBalanced[1, :])

        dcf_list1[0][0] = bayes_risk(None, 0.1, True, True, resultsForUnbalanced[0, :], resultsForUnbalanced[1, :])
        dcf_list1[1][0] = bayes_risk(None, 0.1, True, True, resultsForBalanced[0, :], resultsForBalanced[1, :])

        dcf_list9[0][0] = bayes_risk(None, 0.9, True, True, resultsForUnbalanced[0, :], resultsForUnbalanced[1, :])
        dcf_list9[1][0] = bayes_risk(None, 0.9, True, True, resultsForBalanced[0, :], resultsForBalanced[1, :])
        
        finalMinDCFArray = np.hstack((dcf_list5, dcf_list1))
        finalMinDCFArray = np.hstack((finalMinDCFArray, dcf_list9))
        print("final DCFs for quadratic SVM:")
        print("For Unbalanced:", finalMinDCFArray[0])
        print("For Balanced:", finalMinDCFArray[1])

def main():
    attrs, labels = load('./Train.txt')
    choice = int(input("Type:\n -1 for plotting the raw initial data\n -2 for plotting the gaussianized data\n -3 for the correlation analysis\n -4 for Gaussian models\n -5 for Linear Logistic Regression\n -6 for quad log reg\n -7 for plots for the linear SVM\n -8 for linear SVM\n -9 for plots for the quadratic SVM\n -10 for quad SVM\n"))
    if (choice==1): # Plot original data
        plotInitialData(attrs, labels)
        
    elif(choice==2): # Gaussianize and plot data
        attrs = gaussianize(attrs)
        plotInitialData(attrs, labels)
        
    elif(choice==3): # Correlation analysis
        attrs = gaussianize(attrs) 
        pcc, pcc0, pcc1 = pearson_correlation_coefficient(attrs, labels)
        plotCorrelationHeatMap(pcc, pcc0, pcc1)

    elif(choice==4): # MVG 
        choice2 = int(input("Type:\n -1 for MVG and tied MVG with gaussianization\n -2 for MVG and tied MVG with raw features\n -3 for comparing the covariance matrices for the classes\n"))

        if (choice2==1): # MVG and tied MVG with gaussianization
            choice3 = int(input("-1 to use PCA (m=9)\n-2 to not use PCA\n"))
            if(choice3==1): # Use PCA
                attrs = PCA(attrs, 9)
                k_fold(attrs, labels, 5, 4.1)
            elif(choice3==2): # Dont use PCA
                k_fold(attrs, labels, 5, 4.1)

        elif(choice2==2): # MVG and tied MVG with raw features
            k_fold(attrs, labels, 5, choice)

        elif(choice2==3): # Comparison of cov matrices
            comp_cov_matrix(attrs, labels, True)

    elif(choice==5): # Linear Logistic regression
        choice2 = int(input("Type:\n -1 to plot minDCF without gaussianization\n -2 to plot minDCF with gaussianization\n -3 for log reg without gaussianization\n -4 for log reg with gaussianization\n"))

        if(choice2==1): # plot minDCF without gaussianization
            k_fold(attrs, labels, 5, 5.1)

        elif(choice2==2): # plot minDCF with gaussianization
            # attrs = gaussianize(attrs)
            k_fold(attrs, labels, 5, 5.2)
            
        elif(choice2==3): # log reg without gaussianization
            k_fold(attrs, labels, 5, 5.3)

        elif(choice2==4): # plot minDCF with gaussianization
            k_fold(attrs, labels, 5, 5.4)
            
    elif(choice==6): # Quadratic logistic regression
        choice2 = int(input("Type:\n -1 to plot minDCF without gaussianization\n -2 to plot minDCF with gaussianization\n -3 for quad log reg without gaussianization\n -4 for quad log reg with gaussianization\n"))
        if(choice2==1): # plot minDCF without gaussianization
            k_fold(attrs, labels, 5, 6.1)
        elif(choice2==2):
            k_fold(attrs, labels, 5, 6.2)
        elif(choice2==3):
            k_fold(attrs, labels, 5, 6.3)
        elif(choice2==4):
            k_fold(attrs, labels, 5, 6.4)
    
    elif(choice==7): 
        choice2 = int(input("Type:\n -1 to plot minDCF without gaussianization and without balancing\n -2 to plot minDCF without gaussianization and with balancing\n -3 to plot minDCF with gaussianization and without balancing\n -4 to plot minDCF with gaussianization and with balancing\n"))
        if(choice2==1): # plot minDCF without gaussianization and without balancing
            k_fold(attrs, labels, 5, 7.1)
        elif(choice2==2): # plot minDCF without gaussianization and with balancing
            k_fold(attrs, labels, 5, 7.2)
        elif(choice2==3): # plot minDCF with gaussianization and without balancing
            k_fold(attrs, labels, 5, 7.3)
        elif(choice2==4): # plot minDCF with gaussianization and with balancing
            k_fold(attrs, labels, 5, 7.4)

    elif(choice==8): # Linear SVM with gaussianization and balanced + unbalanced
        k_fold(attrs, labels, 5, choice)

    elif(choice==9): # Quadratic SVM with gaussianization and balanced + unbalanced
        choice2 = int(input("Type:\n -1 to plot minDCF with gaussianization and without balancing\n -2 to plot minDCF with gaussianization and with balancing\n"))
        if(choice2==1): # plot minDCF with gaussianization and without balancing
            k_fold(attrs, labels, 5, 9.1)
        elif(choice2==2): # plot minDCF with gaussianization and with balancing
            k_fold(attrs, labels, 5, 9.2)

    elif(choice==10):
            k_fold(attrs, labels, 5, 10)

    else:
        print("Invalid number")
    

if __name__ == '__main__':
    main()

