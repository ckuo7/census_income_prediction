__author__ = 'chi-liangkuo'




from sklearn.preprocessing import LabelEncoder
from dataProcess import dataProcess
from dataBalance import dataBalance
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit


if __name__ == "__main__":


    np.random.seed(10)

    catList = [1,2,3,4,8,9,12,19,23,32]
    numList = [0,16,18,30,39]

    ##############################################################
    #   Input the training set
    #
    #
    ##############################################################

    df = pd.read_csv('./census-income.data',header=None)
    X_ = dataProcess(df,catList,numList)
    le = LabelEncoder()
    y_ = le.fit_transform(df[41].values)

    new_train_index = dataBalance(y_,0.25)

    # Number of training example:  9976 . Number of positive example:  4988 . Number of negative example:  4988
    # 5%  of training set
    # training set shape   (199523, 228)
    # testing set shape   (99762, 228)
    # training svc with C=10 and gamma =0
    # SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[77939 15637]
    # [  569  5617]]
    # accuracy:  0.8376
    # recall:  0.908
    # precision:  0.2643
    # training svc with C=3 and gamma =0.01
    # SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[78464 15112]
    # [  592  5594]]
    # accuracy:  0.8426
    # recall:  0.9043
    # precision:  0.2702

    # 20% of training example
    # training svc with C=10 and gamma =0
    # SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[78771 14805]
    # [  601  5585]]
    # accuracy:  0.8456
    # recall:  0.9028
    # precision:  0.2739
    # training svc with C=3 and gamma =0.01
    # SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[79362 14214]
    # [  627  5559]]
    # accuracy:  0.8512
    # recall:  0.8986
    # precision:  0.2811

    # Number of training example:  99760 . Number of positive example:  49880 . Number of negative example:  49880
    # 50% of training set
    # training set shape   (199523, 228)
    # testing set shape   (99762, 228)
    # training svc with C=10 and gamma =0
    # SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[78977 14599]
    # [  608  5578]]
    # accuracy:  0.8476
    # recall:  0.9017
    # precision:  0.2765
    # training svc with C=3 and gamma =0.01
    # SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[79644 13932]
    # [  627  5559]]
    # accuracy:  0.8541
    # recall:  0.8986
    # precision:  0.2852

    # # 100% of training example
    # training svc with C=10 and gamma =0
    # ^[[ASVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[79069 14507]
    # [  590  5596]]
    # accuracy:  0.8487
    # recall:  0.9046
    # precision:  0.2784
    # training svc with C=3 and gamma =0.01
    # SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
    # kernel='rbf', max_iter=-1, probability=False, random_state=None,
    # shrinking=True, tol=0.001, verbose=False)
    # predicting...
    # confusion metrics:
    # [[79873 13703]
    # [  624  5562]]
    # accuracy:  0.8564
    # recall:  0.8991
    # precision:  0.2887

    X = X_[new_train_index,:]
    y = y_[new_train_index]

    print "training set shape  ",X_.shape

    ##############################################################
    #   Input the testing set
    #
    #
    ##############################################################

    dft = pd.read_csv('./census-income.test',header=None)
    Xt = dataProcess(dft,catList,numList)
    le2 = LabelEncoder()
    yt = le2.fit_transform(dft[41].values)

    #new_train_index_t = dataBalance(yt,0.01)
    #Xt = Xt[new_train_index_t,:]
    #yt = yt[new_train_index_t]


    print "testing set shape  ",Xt.shape
    print "training svc with C=10 and gamma =0"
    svc1 = SVC(C= 10,kernel='rbf',gamma=0.0)
    svc1.fit(X,y)
    print svc1
    print "predicting..."
    p1 = svc1.predict(Xt)
    m1 = confusion_matrix(yt,p1)
    print "confusion metrics:\n", m1
    print "accuracy: ", round((m1[0,0]+m1[1,1])/float(Xt.shape[0]),4)
    print "recall: ", round(m1[1,1]/float(np.sum(m1[1,:])),4)
    print "precision: ", round(m1[1,1]/float(np.sum(m1[:,1])),4)

    print "training svc with C=3 and gamma =0.01"
    svc2 = SVC(C= 10, kernel='rbf',gamma=0.01)
    svc2.fit(X,y)
    print svc2
    print "predicting..."
    p2 = svc2.predict(Xt)
    m2 = confusion_matrix(yt,p2)
    print "confusion metrics:\n", m2
    print "accuracy: ", round((m2[0,0]+m2[1,1])/float(Xt.shape[0]),4)
    print "recall: ", round(m2[1,1]/float(np.sum(m2[1,:])),4)
    print "precision: ", round(m2[1,1]/float(np.sum(m2[:,1])),4)

    """
    print "training svc with C=0.3 and gamma =0.1"
    svc3 = SVC(C= 0.3, kernel='rbf',gamma=0.1)
    svc3.fit(X,y)
    print "predicting..."
    p3 = svc3.predict(Xt)
    m3 = confusion_matrix(yt,p3)
    print "confusion metrics:\n", m3
    print "accuracy: ", round((m3[0,0]+m3[1,1])/float(Xt.shape[0]),4)
    print "recall: ", round(m3[1,1]/float(np.sum(m3[1,:])),4)
    print "precision: ",round(m3[1,1]/float(np.sum(m3[:,1])),4)
    """







