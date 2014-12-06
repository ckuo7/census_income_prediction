__author__ = 'stevenchu'
import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

import sklearn.cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.metrics import confusion_matrix

from bootstrap import *
from dataBalance import *
from dataProcessOne import *
from dataProcessLabel import *
from dataBalanceRatio import *


# read in data
data = pd.read_csv('census-income.data', header=None)
data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'TARGET']



# this create histograms of values in each column
## good for exploratory purposes
for index in range(len(data.columns)):
    # print Counter(data.ix[:,index].values)
    rcParams.update({'figure.autolayout': True})

    labels, values = zip(*Counter(data.ix[:, index].values).items())
    indexes = np.arange(len(labels))
    width = 1

    plt.barh(indexes, values, width)
    plt.yticks(indexes + width * 0.5, labels)

    plt.savefig('hist_column'+str(index)+'.pdf', format="pdf")

    plt.clf()


# define training and test labels
y = data['TARGET']

# code them as 0s and 1s
# for training and test data
newy = []
for label in y:
    if label == ' 50000+.': newy.append(1)
    else: newy.append(0)

# get rid of labels and BS column within dataset
del data['TARGET']
del data['MARSUPWT']


##### RANDOM FOREST #####
# bootstrapping the smaller class upward, grabbing balanced labels
indices = dataBalanceBoot(newy)

# defining training set for labels by new indices
train_y = y.loc[indices]

# casting as binary variable, so fit and predict methods work better
newy__ = []
for label in train_y:
    if label == ' 50000+.': newy__.append(1)
    else: newy__.append(0)

# defining training set for data by new indices
# processing categorical features
train_data = data.loc[indices]
train_data = dataProcessLabel(train_data)


# investigate rf hyperparameters
# subsetting training data/labels for computation time
## grabbing smaller balanced indices for dataset
train_indices = dataBalance_(newy, 0.01)


# labels subset
train_y = y.loc[train_indices]
newY = []
for label in train_y:
    if label == ' 50000+.': newY.append(1)
    else: newY.append(0)


# data subset
super_train = data.loc[train_indices]
super_train = dataProcessLabel(super_train)


# iterating through various number of trees, various values of features to consider
trees = [10, 50, 100, 250, 500, 1000]
considered = [[4, 'b'], [6, 'g'], [8, 'r'], [10, 'c'], [12, 'm']]


# going through each and creating plots
# different colors are different number of features to consider

# accuracy
for i in considered:
    # set accuracy scores to null list for now
    avg_acc = []
    for j in trees:
        # define RF classifier and fit tree to data
        # append average cross-validation accuracies to empty list
        clf = RandomForestClassifier(n_estimators=j, max_features=i[0], criterion='gini')
        clf.fit(super_train, newY)
        avg_acc.append(np.mean(sklearn.cross_validation.cross_val_score(clf, super_train, newY, cv=5, scoring='accuracy')))
    print trees
    print avg_acc
    plt.plot(trees, avg_acc, i[1], label='%s features' % i[0])

# finally show the resulting graph
plt.title('Accuracy by # Trees Grown and Parameters Chosen', fontsize=20)
plt.ylabel('Accuracy', fontsize=16)
plt.xlabel('Number of Trees Grown', fontsize=16)
plt.legend(loc=4)

plt.show()


# f1
for i in considered:
    # set f1 scores to null list for now
    avg_f1 = []
    for j in trees:
        # define RF classifier and fit tree to data
        # append average cross-validation accuracies to empty list
        clf = RandomForestClassifier(n_estimators=j, max_features=i[0], criterion='gini')
        clf.fit(super_train, newY)
        avg_f1.append(np.mean(sklearn.cross_validation.cross_val_score(clf, super_train, newY, cv=5, scoring='f1')))
    print trees
    print avg_f1
    # plot in specified color
    plt.plot(trees, avg_f1, i[1], label='%s features' % i[0])

# finally show the resulting graph
plt.title('F1 by # Trees Grown and Parameters Chosen', fontsize=20)
plt.ylabel('F1 Score', fontsize=16)
plt.xlabel('Number of Trees Grown', fontsize=16)
plt.legend(loc=4)

plt.show()



##### LOGISTIC REGRESSION #####
# define array of C values to test
cs = [0.0098, 0.013, 0.0173, 0.0229, 0.0303, 0.0402,
     0.0533, 0.0707, 0.0937, 0.1242, 0.1647, 0.2183,
     0.2894, 0.3837, 0.5087, 0.6743, 0.894, 1.1851,
     1.5711, 2.0829, 2.7613, 3.6607, 4.853, 6.4337, 8.5292]


# set up accuracy, precision, recall scores
acc = []
prec = []
f1 = []


super_train_log = data.loc[train_indices]
super_train_log = dataProcessOne(super_train_log)


# iterate through each C value and create logistic regression, populate list with learned parameters
for c in cs:
    # call classifier, fit to training data
    logreg = linear_model.LogisticRegression(C=c, penalty='l1')
    logreg.fit(super_train_log, newY)

    # append average cross-validation scores to accuracy, precision, recall scores
    acc.append(np.mean(sklearn.cross_validation.cross_val_score(logreg, super_train_log, newY, cv=5, scoring='accuracy')))
    prec.append(np.mean(sklearn.cross_validation.cross_val_score(logreg, super_train_log, newY, cv=5, scoring='precision')))
    f1.append(np.mean(sklearn.cross_validation.cross_val_score(logreg, super_train_log, newY, cv=5, scoring='f1')))

print acc
print prec
print f1

# make labels, legends, axes for plots and plot it
plt.plot(cs, acc, 'b', label='Accuracy')
plt.plot(cs, prec, 'r', label='Precision')
plt.plot(cs, f1, 'g', label='F1 Score')
plt.title('Accuracy, Precision, and F1 Score by C values', fontsize=20)
plt.ylabel('Accuracy Metrics', fontsize=16)
plt.xlabel('C values', fontsize=16)
plt.legend(loc=4)

plt.show()