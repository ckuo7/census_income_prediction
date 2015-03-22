__author__ = 'chi-liangkuo'




from sklearn.preprocessing import LabelEncoder
from munge.dataProcess import dataProcess
from munge.dataBalance import dataUnbalance
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import timeit


if __name__ == "__main__":


    np.random.seed(10)
    start = timeit.default_timer()

    catList = [1,2,3,4,8,9,12,19,23,32]
    numList = [0,16,18,30,39]

    ##############################################################
    ###### import the training set
    ###### transform all the categorical variables with one hot
    ###### transformation and standardize the numerical variable
    ###### transform the target variable
    ##############################################################

    df = pd.read_csv('../data/census-income.data',header=None)
    X_ = dataProcess(df,catList,numList)
    le = LabelEncoder()
    y_ = le.fit_transform(df[41].values)

    ##############################################################
    ###### play with stratified K fold
    ##############################################################

    skf = StratifiedKFold(y_,n_folds=5,shuffle=True,random_state=np.random.seed(10) )

    for train_index_s, test_index_s in skf:

        print "length(train_index_s): ",len(train_index_s)
        print "Counter(train_index_s): ",Counter(y_[train_index_s])


        raw_input("press return")


    ##############################################################
    ###### re-balanced the data
    ##############################################################

    # new_train_index = dataBalance(y_,0.01)
    # X = X_[new_train_index,:]
    # y = y_[new_train_index]
    # print "training set shape  ",X_.shape

    ##############################################################
    #######   Input the testing set
    #######   transform the data in the sae way
    ##############################################################

    dft = pd.read_csv('../data/census-income.test',header=None)
    Xt = dataProcess(dft,catList,numList)
    le2 = LabelEncoder()
    yt = le2.fit_transform(dft[41].values)

    ##############################################################
    ###### re-balanced the data
    ##############################################################
    #new_train_index_t = dataUnbalance(yt,0.01,6/float(4))
    #new_train_index_t = dataBalance(yt,0.01)
    #new_train_index_t = dataUnbalance(yt,0.01,7/float(3))
    #new_train_index_t = dataUnbalance(yt,0.01,8/float(2))
    #new_train_index_t = dataUnbalance(yt,0.01,10/float(1))
    #Xt = Xt[new_train_index_t,:]
    #yt = yt[new_train_index_t]

    print "testing set shape  ",Xt.shape




    ##############################################################
    #######   training on the unbalanced data set
    ##############################################################

    # ratio = [ 5.5,5.6, 5.7, 5.8]
    # for r in ratio:
    #
    #     print "####################################################"
    #     print "ratio: ",r
    #     print "####################################################"

    #r = 1
    class_weight = dict()
    class_weight[0] = 1

    #for i in range(1,10):

    # new_train_index = dataUnbalance(y_,0.05,r)
    class_weight[1]=5.5
    new_train_index = dataUnbalance(y_,0.005,1)

    X = X_[new_train_index,:]
    y = y_[new_train_index]

    #print "training set shape  ",X.shape

    #print "training svc with C=10 and gamma =0.01"
    svc2 = SVC(C= 10, kernel='rbf',gamma=0.01, class_weight=class_weight, verbose=False)
    svc2.fit(X,y)
    #print svc2

    # print "saving pickle..."
    # with open('./svc.pkl','wb') as pickle_svc:
    #     cPickle.dump(svc2,pickle_svc)
    print "predicting..."
    p2 = svc2.predict(Xt)


    m2 = confusion_matrix(yt,p2)
    recall2 = m2[1,1]/float(np.sum(m2[1,:]))
    precision2 = m2[1,1]/float(np.sum(m2[:,1]))

    #print "confusion metrics:\n", m2
    print "accuracy: ", round((m2[0,0]+m2[1,1])/float(Xt.shape[0]),4)
    print "f1: ", round( 2*precision2*recall2/(precision2+recall2),4)

    # with open('./svc.pkl','wb') as pickle_svc:
    #     cPickle.dump(svc2,pickle_svc)

    stop = timeit.default_timer()
    print "run time: ",(stop-start)

    ##############################################################
    #######   play with decision function
    ##############################################################

    y_score = svc2.fit(X,y).decision_function(Xt)
    print y_score

    fpr = []
    tpr = []



    fpr, tpr, threshold = roc_curve(yt,y_score)
    roc_auc = auc(fpr,tpr)

    print "fpr: ",fpr
    print "tpr: ",tpr
    print "threshold: ",threshold

    print "roc_auc: ",roc_auc

    plt.figure()
    plt.plot(fpr,tpr,label='ROC curve ( area = 0.2f)' % roc_auc)
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    ##############################################################
    #######   end of play with decision function
    ##############################################################
