from cProfile import label
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from modules.dataTransform import normalize, vrow, vcol, gaussianize, k_folds, PCA
from modules.dataLoad import load, split_db_2to1
from modules.dataEvaluation import logpdf_GAU_ND, empirical_cov, pearson_correlation_coefficient, calculateLogReg, calculateSVM, confusionMatrix, bayes_risk, calc_likehoods_ratio, calc_mu_cov, comp_cov_matrix, log_MVG_Classifier, plotMinDCFLogReg
from modules.dataPlot import plotEstimDensityForRow, plotEstimDensityAllRows, plotInitialData, plotCorrelationHeatMap


def main():
    attrs, labels = load('./Train.txt')
    #plotInitialData(attrs, labels)
    # attrs = normalize(attrs)
    # gaussianized = gaussianize(attrs, True)

    # firstAttr = attrs[1, :]
    # firstAttr = vrow(firstAttr)
    # plotEstimDensityForRow(firstAttr)
    # plotEstimDensityAllRows(attrs)

    # cov, mu = empirical_cov(firstAttr)

    # plt.figure()
    # plt.hist(firstAttr.ravel(), bins=10, density=True)
    # XPlot = np.linspace(-8, 12, 1000)
    # plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, cov)))
    # plt.show()

    #cov,mu = empirical_cov(attrs, True)

    # ====== GAUSSIANIZATION ===================
    # gaussianized = gaussianize(attrs, True)
    # plotInitialData(gaussianized, labels)
    # ==========================================
    
    #pcc0, pcc1 = pearson_correlation_coefficient(attrs, labels)
    #plotCorrelationHeatMap(pcc0)

    # ====== Logistic regression ===============
    (DTR, LTR), (DTE, LTE) = split_db_2to1(attrs, labels) 
    print("Number of samples = 1: ", np.sum(LTR) + np.sum(LTE))
    print("Number of samples ", LTR.shape[0] + LTE.shape[0])
    # predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.3, verbose = True)

    #  Quadratic logistic regression:
    # predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.3, verbose = True, quadratic = True, printIterations=True)
    # ==========================================

    # ================ GMM =====================
    
    gmm = initial_gmm(DTR,LTR)
    opt_gmm = GMM_EM(DTR, gmm, 0.01)
    # gmm_2G = GMM_LBG(opt_gmm, 0.1, 0.01)
    # opt_gmm_2G = GMM_EM(DTR, gmm_2G, 0.01)
    # gmm_4G = GMM_LBG(opt_gmm_2G, 0.1, 0.01)
    # opt_gmm_4G = GMM_EM(DTR, gmm_4G, 0.01)
    # S, logdens = logpdf_GMM(DTE, opt_gmm)
    
    # Prediction
    # JDensities = []
    # for i in range(int(S.shape[0]/2)):
    #     JDensities.append(S[i,:]+S[i+1,:])
    # for i in range(int(S.shape[0]/4)):
    #     JDensities.append(S[i,:]+S[i+1,:]+S[i+2,:]+S[i+3,:])
    # JDensities = np.matrix(S)
    # logSPost = JDensities-logdens
    # prediction = np.argmax(logSPost, axis=0)
    # print(prediction)
    # goodPred = (prediction==LTE).sum()
    # acc = goodPred/LTE.size
    # print(acc)
    
    # ==========================================

    # ================ SVM =====================
    # C = 2
    # predicted = calculateSVM(DTR, LTR, C, DTE, LTE, verbose = False, linear=False, K = 0, gamma = np.exp(-2))
    # print(predicted)
    # confMat = confusionMatrix(predicted, LTE)
    # mu, covMatrix = calc_mu_cov(DTE, LTE)
    # llr = calc_likehoods_ratio(DTE, mu, covMatrix)
    # print("Computed bayes risk (DCF): ", bayes_risk(confMat, 0.5, True, False, llr, LTE, 100))
    # ==========================================

    # ================K-Folds====================
    
    attrs, labels = load('./Train.txt')
    nAttrs = attrs[:, 0:1830]
    nAttrs_PCA = PCA(nAttrs, 10)
    nLabels = labels[0:1830]
    gaussianized_attrs = gaussianize(nAttrs)
    gaussianized_attrs_PCA = PCA(gaussianized_attrs, 10)
    data_folds, labels_folds = k_folds(gaussianized_attrs_PCA, nLabels) # Performs a 10 kfold division
    dcf_list5MVG = []
    dcf_list3 = []
    for i in range(len(data_folds)):
        # Generate the training and testing data and labels
        temp_data = data_folds[:]
        temp_label = labels_folds[:]
        DTE = temp_data.pop(i)
        LTE = temp_label.pop(i)
        temp_data = np.concatenate(temp_data, axis=1)
        temp_label = np.concatenate(temp_label)
        DTR = np.copy(temp_data)
        LTR = np.copy(temp_label)
        
        mu = [] # Training data mean
        cov = [] # Training data cov
        for j in range(np.unique(LTR).size):
            mui = DTR[:,LTR==j].mean(axis=1)
            mui = mui.reshape((mui.shape[0], 1))
            covi, _ = empirical_cov(DTR[:,LTR==j])
            mu.append(mui)
            cov.append(covi)

        # prediction, llr = log_MVG_Classifier(DTE, LTE, mu, cov)
        # minDCF5MVG = bayes_risk(None, 0.5, True, True, llr, LTE, 100)
        # minDCF3 = bayes_risk(None, 0.4, True, True, llr, LTE, 100)
        # dcf_list5MVG.append(minDCF5MVG)
        # dcf_list3.append(minDCF3)

        plotMinDCFLogReg(DTR, LTR, DTE, LTE, 0.5, [10**(-5), 10**2], 20)        

    print("Computed mean bayes risk (DCF) for p = 0.5: ", np.array(dcf_list5MVG).mean())
    print("Computed mean bayes risk (DCF) for p = 0.4: ", np.array(dcf_list3).mean())


if __name__ == '__main__':
    main()

