__author__ = 'chi-liangkuo'


from sklearn.preprocessing import LabelEncoder
from munge.dataProcess import dataProcess
from munge.dataBalance import dataBalance
from munge.dataBalance import dataUnbalance
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import cPickle
import timeit





def knn_test():

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

    df = pd.read_csv('./census-income.data',header=None)
    X_ = dataProcess(df,catList,numList)
    le = LabelEncoder()
    y_ = le.fit_transform(df[41].values)

    r = 5.5
    new_train_index = dataUnbalance(y_,0.05,r)
    X = X_[new_train_index,:]
    y = y_[new_train_index]

    print "training set shape  ",X_.shape

    ##############################################################
    #######   Input the testing set
    #######   transform the data in the sae way
    ##############################################################

    dft = pd.read_csv('./census-income.test',header=None)
    Xt = dataProcess(dft,catList,numList)
    le2 = LabelEncoder()
    yt = le2.fit_transform(dft[41].values)

    # new_train_index_t = dataBalance(yt,0.001)
    # Xt = Xt[new_train_index_t,:]
    # yt = yt[new_train_index_t]


    # print "testing set shape  ",Xt.shape
    # print "training knn with K=50 and uniform weight"
    # knn1 = KNeighborsClassifier(n_neighbors=50,weights='uniform')
    # knn1.fit(X,y)
    # print knn1
    # print "predicting..."
    # p1 = knn1.predict(Xt)
    # m1 = confusion_matrix(yt,p1)
    # print "confusion metrics:\n", m1
    # print "accuracy: ", round((m1[0,0]+m1[1,1])/float(Xt.shape[0]),4)
    # print "recall: ", round(m1[1,1]/float(np.sum(m1[1,:])),4)
    # print "precision: ", round(m1[1,1]/float(np.sum(m1[:,1])),4)

    print "training svc with K=50 and distance weight"
    knn2 = KNeighborsClassifier(n_neighbors=50,weights='distance')
    knn2.fit(X,y)
    print knn2
    # print "saving pickle..."
    # with open('./knn.pkl','wb') as pickle_knn:
    #     cPickle.dump(knn2,pickle_knn)
    print "predicting..."
    p2 = knn2.predict(Xt)
    m2 = confusion_matrix(yt,p2)
    accuracy2 =  round((m2[0,0]+m2[1,1])/float(Xt.shape[0]),4)
    recall2 =  round(m2[1,1]/float(np.sum(m2[1,:])),4)
    precision2 =  round(m2[1,1]/float(np.sum(m2[:,1])),4)

    print "confusion metrics:\n", m2
    print "accuracy: ",accuracy2
    print "f1: ", round( 2*precision2*recall2/(precision2+recall2),4)



    stop = timeit.default_timer()
    print "run time: ",(stop-start)


if __name__ == "__main__":

    knn_test()

    ##############################################################
    #######   final test set result
    ##############################################################
    # testing all testset

    #new_train_index_t = dataBalance(yt,0.05)
    #Xt = Xt[new_train_index_t,:]
    #yt = yt[new_train_index_t]
    #
    # training set shape   (199523, 228)
    # testing set shape   (99762, 228)
    # training svc with C=10 and gamma =0
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #        metric_params=None, n_neighbors=50, p=2, weights='uniform')
    # predicting...
    # confusion metrics:
    # [[74844 18732]
    # [  497  5689]]
    # accuracy:  0.8073
    # recall:  0.9197
    # precision:  0.233
    # training svc with C=3 and gamma =0.01
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #        metric_params=None, n_neighbors=50, p=2, weights='distance')
    # predicting...
    # confusion metrics:
    # [[74792 18784]
    # [  489  5697]]
    # accuracy:  0.8068
    # recall:  0.921
    # precision:  0.2327
    # run time:  983.234070063


    # testing with
    # training set shape   (199523, 228)
    # Number of training example:  9976 . Number of positive example:  4988 . Number of negative example:  4988
    #
    # testing set shape   (9976, 228)
    # training svc with C=10 and gamma =0
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #        metric_params=None, n_neighbors=50, p=2, weights='uniform')
    # predicting...
    # confusion metrics:
    # [[3995  993]
    # [ 398 4590]]
    # accuracy:  0.8606
    # recall:  0.9202
    # precision:  0.8221
    # training svc with C=3 and gamma =0.01
    # KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
    #        metric_params=None, n_neighbors=50, p=2, weights='distance')
    # predicting...
    # confusion metrics:
    # [[3988 1000]
    # [ 392 4596]]
    # accuracy:  0.8605
    # recall:  0.9214
    # precision:  0.8213
    # run time:  119.611333132