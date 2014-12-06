__author__ = 'chi-liangkuo'

import timeit
import numpy as np
import pandas as pd
import chooseFeature
import operator

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
from collections import Counter
from dataProcess import dataProcess
from dataBalance import dataBalance
from bootstrap import dataBalanceBoot


def main():

    df = pd.read_csv('./census-income.data',header=None)
    #l = [0,2,3,4,5,7,8,9,10,11,12,16,18,30,31,32,33,34,35]
    #l = [0,1,2,3,4,5,6,9,11,13,16,17,20,24,30,32,35,36]
    l = [0,4,30,31,39]
    X_ = dataProcess(df[l])
    le = LabelEncoder()
    y_ = le.fit_transform(df[41].values)

    #new_train_index = dataBalance(y_,0.05)
    new_train_index = dataBalanceBoot(y_)
    X = X_[new_train_index,:]
    y = y_[new_train_index]


    clf = SVC()
    #clf.fit(X_train,y_train)
    #score = clf.score(X_test,y_test)
    #print score
    a = cross_val_score(clf,X,y,cv=3,scoring="accuracy")
    print "accuracy with 3 folds cross validation: ",np.mean(a)
    p = cross_val_score(clf,X,y,cv=3,scoring="precision")
    print "precision with 3 folds cross validation: ",np.mean(p)
    r = cross_val_score(clf,X,y,cv=3,scoring="recall")
    print "recall with 3 folds cross validation: ",np.mean(r)
    #r2 = cross_val_score(clf,X_[new_train_index,:],y[new_train_index],cv=3,scoring="roc_auc")


    #    ,np.mean(r2)


    #print df[l]
    #featureReport(df)
    #clf =chooseFeature.chooseFeature()
    #le = LabelEncoder()


    # new_index = dataBalanceBoot(le.fit_transform(df[41].values))
    # newdf = df.ix[new_index,:]
    # impurity = clf.ftr_seln(newdf[l],newdf[41].values)
    #
    # for k,v in  Counter(impurity).items():
    #     print k,v
    # # sorted_x = sorted(impurity.items(), key=operator.itemgetter(1))
    # for k,v in sorted_x.items():
    #     print k,v

    #X_ = dataProcess(df[l])

    #X = df[range(41)]
    #print X_.shape

    #le = LabelEncoder()
    #y = le.fit_transform(df[41].values)
    #nrow = y.shape[0]

    #training_size = 0.05

    #new_train_index = dataBalance(y,0.005)
    #new_train_index = dataBalance(y,0.005)

    #holdout_index = set(np.arange(nrow)).difference(train_index)
    #test_index = np.random.choice(list(holdout_index),nrow*0.02,replace=False)



    #print len(new_train_index)
    # X_train = X_[new_train_index,:]
    # y_train = y[new_train_index]
    #
    # print Counter(y_train)

    #X_test = X_[test_index,:]
    #y_test = y[test_index]



if __name__ == "__main__":
    start = timeit.default_timer()
    main()
    stop = timeit.default_timer()
    print "run time: ",(stop-start)