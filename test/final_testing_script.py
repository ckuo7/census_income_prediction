__author__ = 'chi-liangkuo'

import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from munge.dataProcess import dataProcess
from sklearn.metrics import confusion_matrix
from logistic_test import logistic_test
from rf_testing import rf_testing
from knn_test import knn_test

def final_testing():

    np.random.seed(10)
    catList = [1,2,3,4,8,9,12,19,23,32]
    numList = [0,16,18,30,39]

    dft = pd.read_csv('../data/census-income.test',header=None)
    Xt = dataProcess(dft,catList,numList)
    le2 = LabelEncoder()
    yt = le2.fit_transform(dft[41].values)

    print "######################################################################"
    print "######## Testing the SVC"
    print "######################################################################"

    with open('../pkl/svc.pkl','rb') as svc_pickle:

        svc = pickle.load(svc_pickle)
        print svc
        print svc.support_


        print "predicting..."
        p2 = svc.predict(Xt)
        m2 = confusion_matrix(yt,p2)
        recall2 = m2[1,1]/float(np.sum(m2[1,:]))
        precision2 = m2[1,1]/float(np.sum(m2[:,1]))

        print "confusion metrics:\n", m2
        print "accuracy: ", round((m2[0,0]+m2[1,1])/float(Xt.shape[0]),4)
        print "f1:       ", round( 2*precision2*recall2/(precision2+recall2),4)

    print "######################################################################"
    print "######## Testing the KNN"
    print "######################################################################"


    knn_test()


    print "######################################################################"
    print "######## Testing the choose feature"
    print "######################################################################"


    with open('../pkl/chooseFeature.pkl','rb') as cf_pickle:

        cf = pickle.load(cf_pickle)
        print cf
        print "predicting..."
        p2 = cf.predict(Xt)
        m2 = confusion_matrix(yt,p2)
        recall2 = m2[1,1]/float(np.sum(m2[1,:]))
        precision2 = m2[1,1]/float(np.sum(m2[:,1]))

        print "confusion metrics:\n", m2
        print "accuracy: ", round((m2[0,0]+m2[1,1])/float(Xt.shape[0]),4)
        print "f1:       ", round( 2*precision2*recall2/(precision2+recall2),4)


    print "######################################################################"
    print "######## Testing the Logistic Regression"
    print "######################################################################"

    logistic_test()

    print "######################################################################"
    print "######## Testing the Random Forest"
    print "######################################################################"

    rf_testing()

if __name__ == "__main__":
    final_testing()