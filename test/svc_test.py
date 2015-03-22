__author__ = 'chi-liangkuo'




from sklearn.preprocessing import LabelEncoder
from munge.dataProcess import dataProcess
from munge.dataBalance import dataBalance
from munge.dataBalance import dataUnbalance
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

import pandas as pd
import numpy as np
import cPickle
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
    ###### testing on the final model
    ##############################################################

    # print "training svc with C=10 and gamma =0"
    # svc1 = SVC(C= 10,kernel='rbf',gamma=0.0)
    # svc1.fit(X,y)
    # print svc1
    # print "predicting..."
    # p1 = svc1.predict(Xt)
    # m1 = confusion_matrix(yt,p1)
    # print "confusion metrics:\n", m1
    # print "accuracy: ", round((m1[0,0]+m1[1,1])/float(Xt.shape[0]),4)
    # print "recall: ", round(m1[1,1]/float(np.sum(m1[1,:])),4)
    # print "precision: ", round(m1[1,1]/float(np.sum(m1[:,1])),4)
    #
    # print "training svc with C=3 and gamma =0.01"
    # svc2 = SVC(C= 10, kernel='rbf',gamma=0.01)
    # svc2.fit(X,y)
    # print svc2
    # print "predicting..."
    # p2 = svc2.predict(Xt)
    # m2 = confusion_matrix(yt,p2)
    # print "confusion metrics:\n", m2
    # print "accuracy: ", round((m2[0,0]+m2[1,1])/float(Xt.shape[0]),4)
    # print "recall: ", round(m2[1,1]/float(np.sum(m2[1,:])),4)
    # print "precision: ", round(m2[1,1]/float(np.sum(m2[:,1])),4)


    # print "training svc with C=0.3 and gamma =0.1"
    # svc3 = SVC(C= 0.3, kernel='rbf',gamma=0.1)
    # svc3.fit(X,y)
    # print "predicting..."
    # p3 = svc3.predict(Xt)
    # m3 = confusion_matrix(yt,p3)
    # print "confusion metrics:\n", m3
    # print "accuracy: ", round((m3[0,0]+m3[1,1])/float(Xt.shape[0]),4)
    # print "recall: ", round(m3[1,1]/float(np.sum(m3[1,:])),4)
    # print "precision: ",round(m3[1,1]/float(np.sum(m3[:,1])),4)
    #


    ##############################################################
    #######   training on the unbalanced data set
    ##############################################################

    # ratio = [ 5.5,5.6, 5.7, 5.8]

    # for r in ratio:
    #
    #     print "####################################################"
    #     print "ratio: ",r
    #     print "####################################################"

    r = 1
    class_weight = {1:1,0:5.5}

    # new_train_index = dataUnbalance(y_,0.05,r)
    new_train_index = dataUnbalance(y_,0.001,r)

    X = X_[new_train_index,:]
    y = y_[new_train_index]

    print "training set shape  ",X.shape

    print "training svc with C=10 and gamma =0.01"
    svc2 = SVC(C= 10, kernel='rbf',gamma=0.01, class_weight=class_weight, verbose=True)
    svc2.fit(X,y)
    print svc2

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
    #######   play with decision function
    ##############################################################

    # print "saving pickle..."
    # with open('./svc.pkl','wb') as pickle_svc:
    #     cPickle.dump(svc2,pickle_svc)
    print "predicting..."
    p2 = svc2.predict(Xt)


    m2 = confusion_matrix(yt,p2)
    recall2 = m2[1,1]/float(np.sum(m2[1,:]))
    precision2 = m2[1,1]/float(np.sum(m2[:,1]))

    print "confusion metrics:\n", m2
    print "accuracy: ", round((m2[0,0]+m2[1,1])/float(Xt.shape[0]),4)
    print "f1: ", round( 2*precision2*recall2/(precision2+recall2),4)

    with open('./svc.pkl','wb') as pickle_svc:
        cPickle.dump(svc2,pickle_svc)

    stop = timeit.default_timer()
    print "run time: ",(stop-start)

######################################################
###### ratio results on svc with gamma = 0.01 C= 10
######################################################

# Last login: Wed Dec  3 15:44:07 on ttys004
# chi-liangkuo@chi-liangs-air:~/Desktop/module2/MSAN_621/project$ python svc_test.py
# testing set shape   (99762, 228)
# ####################################################
# ratio:  1.5
# ####################################################
# Number of training example:  4987 . Number of positive example:  1995 . Number of negative example:  2992
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[82477 11099]
#  [  926  5260]]
# accuracy:  0.8795
# f1:  0.4666
# ####################################################
# ratio:  2.33333333333
# ####################################################
# Number of training example:  6650 . Number of positive example:  1995 . Number of negative example:  4655
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[85772  7804]
#  [ 1297  4889]]
# accuracy:  0.9088
# f1:  0.5179
# ####################################################
# ratio:  4.0
# ####################################################
# Number of training example:  9975 . Number of positive example:  1995 . Number of negative example:  7980
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[89580  3996]
#  [ 2152  4034]]
# accuracy:  0.9384
# f1:  0.5675
# ####################################################
# ratio:  5
# ####################################################
# Number of training example:  11970 . Number of positive example:  1995 . Number of negative example:  9975
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[90432  3144]
#  [ 2421  3765]]
# accuracy:  0.9442
# f1:  0.575

# ####################################################
# ratio:  5.3
# ####################################################
# Number of training example:  12568 . Number of positive example:  1995 . Number of negative example:  10573
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[90812  2764]
#  [ 2555  3631]]
# accuracy:  0.9467
# f1:  0.5772
# ####################################################
# ratio:  5.5
# ####################################################
# Number of training example:  12967 . Number of positive example:  1995 . Number of negative example:  10972
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[90567  3009]
#  [ 2446  3740]]
# accuracy:  0.9453
# f1:  0.5783

# ####################################################
# ratio:  5.6
# ####################################################
# Number of training example:  13167 . Number of positive example:  1995 . Number of negative example:  11172
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[90999  2577]
#  [ 2660  3526]]
# accuracy:  0.9475
# f1:  0.5738
# ####################################################
# ratio:  5.7
# ####################################################
# Number of training example:  13366 . Number of positive example:  1995 . Number of negative example:  11371
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[90963  2613]
#  [ 2609  3577]]
# accuracy:  0.9477
# f1:  0.5781
# ####################################################
# ratio:  5.8
# ####################################################
# Number of training example:  13566 . Number of positive example:  1995 . Number of negative example:  11571
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[91011  2565]
#  [ 2628  3558]]
# accuracy:  0.9479
# f1:  0.5781

# ####################################################
# ratio:  6
# ####################################################
# Number of training example:  13965 . Number of positive example:  1995 . Number of negative example:  11970
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[91089  2487]
#  [ 2699  3487]]
# accuracy:  0.948
# f1:  0.5735
# ####################################################
# ratio:  7
# ####################################################
# Number of training example:  15960 . Number of positive example:  1995 . Number of negative example:  13965
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[91687  1889]
#  [ 2960  3226]]
# accuracy:  0.9514
# f1:  0.5709

# ####################################################
# ratio:  10.0
# ####################################################
# Number of training example:  21945 . Number of positive example:  1995 . Number of negative example:  19950
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[92575  1001]
#  [ 3672  2514]]
# accuracy:  0.9532
# f1:  0.5183
# ####################################################
# ratio:  15.0
# ####################################################
# Number of training example:  31920 . Number of positive example:  1995 . Number of negative example:  29925
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[93148   428]
#  [ 4530  1656]]
# accuracy:  0.9503
# f1:  0.4005
# ####################################################
# ratio:  20
# ####################################################
# Number of training example:  41895 . Number of positive example:  1995 . Number of negative example:  39900
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[93377   199]
#  [ 4991  1195]]
# accuracy:  0.948
# f1:  0.3153
# ####################################################
# ratio:  50
# ####################################################
# Number of training example:  101745 . Number of positive example:  1995 . Number of negative example:  99750
#
# training svc with C=10 and gamma =0.01
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.01,
#   kernel='rbf', max_iter=-1, probability=False, random_state=None,
#   shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[93530    46]
#  [ 5553   633]]
# accuracy:  0.9439
# f1:  0.1844



# ####################################################
# accuracy with size of training set
# ####################################################

# test unbalance data set

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

# Number of training example:  371112 . Number of positive example:  185556 . Number of negative example:  185556
# training with 186 percent
# training set shape   (199523, 228)
# testing set shape   (99762, 228)
# training svc with C=10 and gamma =0
# SVC(C=10, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
# kernel='rbf', max_iter=-1, probability=False, random_state=None,
# shrinking=True, tol=0.001, verbose=False)
# predicting...
# confusion metrics:
# [[79345 14231]
# [  613  5573]]
# accuracy:  0.8512
# recall:  0.9009
# precision:  0.2814
