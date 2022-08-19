# Wine quality project - Machine Learning 2022

## Goal
Dataset contains 10 classes regarging the quality of thw wines and has been binarized: lower than 6 = 0 and greater than 6 = 1. Wines with quality 6 have been discarded.

11 features in total (attributes)

Classes are hard to separate, expect higher error rates (in the order of ten percent).

---
## Description of the dataset

Attribute Information:

For more information, read [Cortez et al., 2009].
Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulphates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10)

## Transformation of data

### K-folds approach

We choose to use the k-fold approach since the dataset is considered to be small sized. In the case the k-fold takes too long to perform all the calculations, then the single-fold approach could be used.

Total amount of samples = 1839. With the algorithm written, each partition will have 1839/5 = 367.8 which is rounded to 367. For the last partition, it will have 371.
Picked the avg of the minDCF since k-fold allows observing how different values could be obtained with different combination of data. 
Data was mixed once, randomly, at the beginning of the program to make the partitions more homogeneous. 

## Analisis of data

### Features

The following images are the original shape of the data without any tranformation. The raw features presnt irregular distributions and some of them show outliners, which lead to expect sub-optimal results with Gaussian based methods.
Processing of the data such as gausseanization are expected to improve the results.