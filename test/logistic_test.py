__author__ = 'stevenchu'

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import pandas as pd
from collections import Counter
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from munge.dataProcessOne import *
from munge.dataBalanceRatio import *



def logistic_test():
    np.random.seed(10)

    data = pd.read_csv('census-income.data', header=None)
    data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                    'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                    'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                    'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                    'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                    'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                    'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'TARGET']


    test_data = pd.read_csv('census-income.test', header=None)
    test_data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                    'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                    'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                    'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                    'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                    'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                    'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'TARGET']


    # define training and test labels
    y = data['TARGET']
    test_y = test_data['TARGET']


    # THIS GOOD FOR FINDING AND GETTING RID OF SHIT NOT IN TESTING DATA
    ind = data.loc[data['HHDFMX'] == ' Grandchild <18 ever marr not in subfamily']
    data = data[data['HHDFMX'] != ' Grandchild <18 ever marr not in subfamily']
    for i in ind.index:
        del y[i]


    # cleaning up data even more
    del data['MARSUPWT']
    del data['TARGET']
    del test_data['MARSUPWT']
    del test_data['TARGET']


    # code them as 0s and 1s, for training and test data
    train_newy = []
    for label in y:
        if label == ' 50000+.': train_newy.append(1)
        else: train_newy.append(0)

    test_newy = []
    for label in test_y:
        if label == ' 50000+.': test_newy.append(1)
        else: test_newy.append(0)


    data = dataProcessOne(data)
    test_data = dataProcessOne(test_data)


    # ratios = [float(2)/1, float(3)/1, float(4)/1, float(5)/1, float(6)/1, float(7)/1, float(8)/1, float(9)/1, float(10)/1, float(11)/1, float(12)/1, float(13)/1, float(14)/1, float(15)/1]
    ratios = [float(2)/1]

    for r in ratios:

        train_index = dataUnbalance(train_newy, 1.0, r)
        #train_index = dataUnbalance(train_newy, 0.0001, r)
        # defining training set for labels by new indices
        train_y = y.loc[train_index]

        # casting as binary variable, so fit and predict methods work better
        newY = []
        for label in train_y:
            if label == ' 50000+.': newY.append(1)
            else: newY.append(0)

        # defining training set for data by new indices
        # processing categorical features
        train_data = data[train_index, :]

        # initializing and fitting classifier algorithm
        logreg = linear_model.LogisticRegression(C=0.3837, penalty='l1')
        logreg.fit(train_data, newY)

        # getting predictions and generating confusion matrix
        predictions = logreg.predict(test_data)
        cm = confusion_matrix(test_newy, predictions)

        print 'predictions:', Counter(predictions)
        print 'actual:', Counter(test_newy)
        print '\n'

        cm = confusion_matrix(test_newy, predictions)

        accuracy = float(cm[0][0]+cm[1][1])/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0])
        print 'acc: ', accuracy
        precision = float(cm[0][1])/(cm[0][1]+cm[1][1])
        print 'prec: ', precision
        recall = float(cm[1][0])/(cm[1][0]+cm[1][1])
        print 'recall: ', recall
        f = precision*recall*2/(precision+recall)

        print 'ratio: ', str(r), ',', 'f score: ', f

        # with open('logreg.pkl', 'wb') as pickle_rf:
        #     pickle.dump(logreg, pickle_rf)

if __name__ == "__main__":
    logistic_test()