# Wine quality project - Machine Learning 2022

- Pier Luigi Nakai Ricchetti - s293742
- Juan José Jaramillo Botero - s301352

---
## Description of the dataset
The dataset contains 10 classes regarging the quality of the wines and has been binarized: lower than 6 = 0 and greater than 6 = 1. Wines with quality 6 have been discarded.

11 features in total (attributes)

Classes are hard to separate, expect higher error rates (in the order of ten percent).

### Attribute Information:

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

## Code info

The main application code is on the file "projectCode.py". It is ready to be ran and it will display a simple menu where the user can choose what to execute.
The number to be taken as input might have a "sub-menu" where other correlated actions can be performed. Everything reported on the pdf, whether it is an image or just number results, can be ran again using this menu. 

It is important to notice, however, that the results may fluctuate a little bit, since the k-fold approach performs a random shuffle on the dataset that is different everytime it is ran. 

## More information

For the report and detailed information, please, check the pdf report. 
