import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import numpy as np
from .dataEvaluation import logpdf_GAU_ND, empirical_cov, bayes_risk
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
        #print(D0)
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
            plt.hist(D0[i, :], bins = 30, density = True, alpha = 0.4, label = "bad wines")
            plt.hist(D1[i, :], bins = 30, density = True, alpha = 0.4, label = "good wines")
            plt.legend()
            plt.tight_layout()
            # plt.savefig('images/gaussianized/' + attrList[i] + '.png')
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
    
def plotCorrelationHeatMap(raw, class1, class2):
    '''
        Plots a heat map to analize the correlation between attributes
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
    # plt.figure()
    fig, ax = plt.subplots()
    heatmap = ax.imshow(raw, vmin=-1, vmax=1, cmap='inferno')
    fig.colorbar(heatmap)
    # ax.set_xticks(np.arange(len(attrList)), labels=attrList)
    # ax.set_yticks(np.arange(len(attrList)), labels=attrList)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.xlabel("Raw Data")

    fig1, ax1 = plt.subplots()
    heatmap1 = ax1.imshow(class1, vmin=-1, vmax=1, cmap='RdBu')
    fig1.colorbar(heatmap1)
    # ax.set_xticks(np.arange(len(attrList)), labels=attrList)
    # ax.set_yticks(np.arange(len(attrList)), labels=attrList)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.xlabel("Class One")

    fig2, ax2 = plt.subplots()
    heatmap2 = ax2.imshow(class2, vmin=-1, vmax=1, cmap='YlGn')
    fig2.colorbar(heatmap2)
    # ax.set_xticks(np.arange(len(attrList)), labels=attrList)
    # ax.set_yticks(np.arange(len(attrList)), labels=attrList)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    plt.xlabel("Class Two")

    # plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.show()
    return

# ===================================== Bayes plt =================================

def confusion_matrix(prediction, correct):
    '''
    ## Params:
    - prediction = List of predicted classes (assumed to be an 1 dim np array)
    - correct = List of correct classes
    '''
    # Creating empty confusion matrix
    confMatrix = np.zeros((2, 2))
    prediction = prediction.astype(int)
    correct = correct.astype(int)

    for i in range (prediction.shape[0]):
        confMatrix[prediction[i]][correct[i]] += 1
    return confMatrix

def calculateRoc(resolution, scores, labels, plotLabels):
    
    '''
    ## Explanation:
    The ideia is to get a list of thresholds and plotting a curve FPR x TPR (where TPR = 1 - FNR)

    ## Params:
    - resolution = number of steps to calculate threshold (the resolution of the graph)
    - scores = Scores to be plotted (N, labels.shape[0]) where N is the number of scores
    - labels = correct list of labels 
    '''
    scoresNumber = scores.shape[0]

    FPRvalues = np.zeros((scoresNumber, 0))
    FNRvalues = np.zeros((scoresNumber, 0))

    maxScores = np.max(scores, 1)
    minScores = np.min(scores, 1)

    tempFNR = np.zeros((scoresNumber, 1))
    tempFPR = np.zeros((scoresNumber, 1))

    for t in range (1, resolution, 1):
        for i in range(scoresNumber):

            maxThresh = maxScores[i]
            minThresh = minScores[i]

            thresh = t*( (maxThresh-minThresh)/resolution )
            thresh += minThresh
            predicted = np.where(scores[i] > thresh, 1, 0)
            confMatrix = confusion_matrix(predicted, labels)

            # Retrieve number of false negatives and false positives
            FN = confMatrix[0][1]
            FP = confMatrix[1][0]
            # print("FP = ", FP)
            # print("FN = ", FN)

            # Retrieving the true positive and true negative values:
            TP = confMatrix[1][1]
            TN = confMatrix[0][0]
            # print("TN = ", TN)

            FNR = FN/(FN + TP)
            FPR = FP/(FP + TN)

            tempFNR[i][0] = FNR
            tempFPR[i][0] = FPR

        FNRvalues = np.hstack((FNRvalues, tempFNR))
        FPRvalues = np.hstack((FPRvalues, tempFPR))
        

    plt.figure()
    for i in range(scoresNumber):
        plt.plot(FPRvalues[i], FNRvalues[i], label = plotLabels[i])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("FNR")
    plt.title("DET curve")
    plt.show()

def bayes_error_plot(scores, LTE, labelsForScores, pRange = (-3, 3, 21), fusion = False):
    '''
    ## Ideia:
    For each p, will calculate the normalized DCF using pi = 1/( 1+e^(-p) ) by using the bayes decision for (pi, 1, 1)
    ## Params:
    - scores = (2, N) each line corresponding to one score to be calculated. If fusion: (3, N) where the last line is the fusion model.
    - labelsForScores = Labels to be put in each score to be plotted ["", ""]
    - pRange = (minRange, maxRAnge, step)
    - fusion = if want to evaluate for the fusion model.... The logic of the plot is different. IN scores, fusion model should be the last line.
    '''
    if (fusion == False):
        effPriorLogOdds = np.linspace(pRange[0], pRange[1], pRange[2])

        DCFValues = np.zeros((2,0))
        minDCFValues = np.zeros((2,0))

        for i in range(effPriorLogOdds.shape[0]):

            # Retrieving p:
            p = effPriorLogOdds[i]

            # Calculating effective prior piTil:
            piTil = 1/( 1 + np.exp(-p) )

            # Calculating the prediction for pi and its confusion matrix:
            # confMatrix = confusion_matrix(prediction, LTE)

            # Retrieving normalized DCF:
            DCF = bayes_risk(None, piTil, True, False, scores[0], LTE, threshold = -p)
            DCF = np.vstack((DCF, bayes_risk(None, piTil, True, False, scores[1], LTE, threshold = -p)))
            DCFValues = np.hstack((DCFValues, DCF))

            # Retrieving minimum normalized DCF:
            minDCF = bayes_risk(None, piTil, True, True, scores[0], LTE)
            minDCF = np.vstack((minDCF, bayes_risk(None, piTil, True, True, scores[1], LTE)))
            minDCFValues = np.hstack((minDCFValues, minDCF))

        plt.figure()
        plt.plot(effPriorLogOdds, DCFValues[0], label="DCF - " + labelsForScores[0], color="r")
        plt.plot(effPriorLogOdds, DCFValues[1], label="DCF - " + labelsForScores[1], color="b")
        plt.plot(effPriorLogOdds, minDCFValues[0], label="min DCF - " + labelsForScores[0], color="r", linestyle = 'dashed')
        plt.plot(effPriorLogOdds, minDCFValues[1], label="min DCF - " + labelsForScores[1], color="b", linestyle = 'dashed')
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.xlabel(r"Prior log odds = $log\frac{\tilde \pi}{1 - \tilde \pi}$")
        plt.ylabel("DCF")
        plt.legend()
        plt.show()
        plt.close()

    else:
        effPriorLogOdds = np.linspace(pRange[0], pRange[1], pRange[2])

        DCFValues = np.zeros((3,0))
        minDCFValues = np.zeros((1,0))

        for i in range(effPriorLogOdds.shape[0]):

            # Retrieving p:
            p = effPriorLogOdds[i]

            # Calculating effective prior piTil:
            piTil = 1/( 1 + np.exp(-p) )

            # Calculating the prediction for pi and its confusion matrix:
            # confMatrix = confusion_matrix(prediction, LTE)

            # Retrieving normalized DCF:
            DCF = bayes_risk(None, piTil, True, False, scores[0], LTE, threshold = -p)
            DCF = np.vstack((DCF, bayes_risk(None, piTil, True, False, scores[1], LTE, threshold = -p)))
            DCF = np.vstack((DCF, bayes_risk(None, piTil, True, False, scores[2], LTE, threshold = -p)))
            DCFValues = np.hstack((DCFValues, DCF))

            # Retrieving minimum normalized DCF for the Fusion model:
            minDCF = np.reshape(bayes_risk(None, piTil, True, True, scores[2], LTE), (1, 1))
            minDCFValues = np.hstack((minDCFValues, minDCF))

        plt.figure()
        plt.plot(effPriorLogOdds, DCFValues[0], label="act DCF - " + labelsForScores[0], color="r")
        plt.plot(effPriorLogOdds, DCFValues[1], label="act DCF - " + labelsForScores[1], color="b")
        plt.plot(effPriorLogOdds, DCFValues[2], label="act DCF - " + labelsForScores[2], color="g")
        plt.plot(effPriorLogOdds, minDCFValues[0], label="min DCF - " + labelsForScores[2], color="g", linestyle = 'dashed')
        plt.ylim([0, 1.1])
        plt.xlim([-3, 3])
        plt.xlabel(r"Prior log odds = $log\frac{\tilde \pi}{1 - \tilde \pi}$")
        plt.ylabel("DCF")
        plt.legend()
        plt.show()
        plt.close()
