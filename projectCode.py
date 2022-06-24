from cProfile import label
from random import gauss
import numpy as np
import matplotlib.pyplot as plt
from modules.dataTransform import normalize, vrow, vcol, gaussianize
from modules.dataLoad import load
from modules.dataEvaluation import comp_cov_matrix, logpdf_GAU_ND, empirical_cov, pearson_correlation_coefficient
from modules.dataPlot import plotEstimDensityForRow, plotEstimDensityAllRows, plotInitialData, plotCorrelationHeatMap

def main():
    attrs, labels = load('./Train.txt')
    #plotInitialData(attrs, labels)
    attrs = normalize(attrs)
    gaussianized = gaussianize(attrs)

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
    
    # plotInitialData(gaussianized, labels)
    
    # check the gaussean model
    # pcc0, pcc1 = pearson_correlation_coefficient(gaussianized, labels);
    # plotCorrelationHeatMap(pcc0)
    # plotCorrelationHeatMap(pcc1) 
    # comp_cov_matrix(gaussianized, labels, True)
    
if __name__ == '__main__':
    main()

