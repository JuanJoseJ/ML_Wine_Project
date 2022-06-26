from cProfile import label
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from modules.dataTransform import normalize, vrow, vcol, gaussianize
from modules.dataLoad import load, split_db_2to1
from modules.dataEvaluation import logpdf_GAU_ND, empirical_cov, pearson_correlation_coefficient, calculateLogReg, calculateSVM, confusionMatrix, bayes_risk, calc_likehoods_ratio, calc_mu_cov, comp_cov_matrix
from modules.dataPlot import plotEstimDensityForRow, plotEstimDensityAllRows, plotInitialData, plotCorrelationHeatMap



def main():
    attrs, labels = load('./Train.txt')
    #plotInitialData(attrs, labels)
    attrs = normalize(attrs)
    gaussianized = gaussianize(attrs, True)

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
    # predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.3, verbose = True)

    #  Quadratic logistic regression:
    # predicted = calculateLogReg(DTR, LTR, DTE, LTE, 0.3, verbose = True, quadratic = True, printIterations=True)
    # ==========================================

    # ================ SVM =====================
    C = 2
    predicted = calculateSVM(DTR, LTR, C, DTE, LTE, verbose = False, linear=False, K = 0, gamma = np.exp(-2))
    print(predicted)
    confMat = confusionMatrix(predicted, LTE)
    mu, covMatrix = calc_mu_cov(DTE, LTE)
    llr = calc_likehoods_ratio(DTE, mu, covMatrix)
    print("Computed bayes risk (DCF): ", bayes_risk(confMat, 0.5, True, False, llr, LTE, 100))
    # ==========================================

if __name__ == '__main__':
    main()

