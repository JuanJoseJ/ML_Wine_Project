import numpy as np
import matplotlib.pyplot as plt

from modules.dataTransform import normalize, empirical_cov, plotInitialData, logpdf_GAU_ND, vrow, vcol, plotEstimDensityForRow, plotEstimDensityAllRows
from modules.dataLoad import load


def main():
    attrs, labels = load('./Train.txt')
    #plotInitialData(attrs, labels)
    #attrs = normalize(attrs)

    firstAttr = attrs[1, :]
    firstAttr = vrow(firstAttr)
    plotEstimDensityForRow(firstAttr)
    plotEstimDensityAllRows(attrs)

    # cov, mu = empirical_cov(firstAttr)

    # plt.figure()
    # plt.hist(firstAttr.ravel(), bins=10, density=True)
    # XPlot = np.linspace(-8, 12, 1000)
    # plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, cov)))
    # plt.show()

    #cov,mu = empirical_cov(attrs, True)



if __name__ == '__main__':
    main()

