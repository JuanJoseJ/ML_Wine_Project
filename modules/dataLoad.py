import numpy as np
from .dataTransform import vcol

def load(address):
    '''
    Load the dataset into a matrix with dimension MxN where M = attributes and N = different samples
    Line retrived from dataset:
    8.1, 0.27, 0.41, 1.45, 0.033, 11,63,0.9908,2.99,0.56,12,0
    8.2, 0.27, 0.41, 1.45, 0.033, 11,63,0.9908,2.99,0.56,12,0
    '''
    dataList = []
    labelList = []
    with open(address) as text:
        for line in text:
            attrs = line.split(',')[0:11]
            attrs = vcol(np.array([float(i) for i in attrs]))
            label = line.split(',')[-1].strip()
            dataList.append(attrs)
            labelList.append(label)
    return np.hstack(dataList), np.array(labelList)
