import numpy as np

def vcol(vlist):

    '''
    Will transform a 1D list into a column vector (rows = len(vlist) // columns = 1)
     1d list = [1,2,3] --vcol--> 
     
     [[1],
     [2],
     [3]]
    '''
    return np.reshape(vlist, (len(vlist), 1))

def vrow(vlist):
    '''
     Will transform a 1D list into a row vector (rows = 1 // columns = len(vlist))
     1d list = [1,2,3] --vrow--> [[1,2,3]]
    '''
    return np.reshape(vlist, (1, len(vlist)))

def load(address):
    '''
    Load the dataset into a matrix with dimension MxN where M = attributes and N = different samples
    Line reeived from dataset:
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
            print()
    return np.hstack(dataList), np.array(labelList)


def main():
    JuanFresco = True
    print("is Juan Fresco? ", JuanFresco)

if __name__ == '__main__':
    main()

