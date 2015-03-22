__author__ = 'chi-liangkuo'


from sklearn.preprocessing import LabelEncoder
from munge.dataBalance import dataBalance
from model.chooseFeature import chooseFeature
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import pickle
import timeit



if __name__ == "__main__":


    df = pd.read_csv('./census-income.data',header=None)
    le = LabelEncoder()
    y = le.fit_transform(df[41].values)
    new_index = dataBalance(y,0.93)

    ##############################################################
    #   Input the testing set
    #
    #
    ##############################################################

    dft = pd.read_csv('./census-income.test',header=None)
    Xt = dft.values
    le2 = LabelEncoder()
    yt = le2.fit_transform(dft[41].values)

    new_train_index_t = dataBalance(yt,0.025)
    Xt = Xt[new_train_index_t,:]
    yt = yt[new_train_index_t]

    clf = chooseFeature()
    clf.fit(df.values,y)
    p1 = clf.predict(Xt)
    m1 = confusion_matrix(yt,p1)
    print "confusion metrics:\n", m1
    print "accuracy: ", round((m1[0,0]+m1[1,1])/float(Xt.shape[0]),4)
    print "recall: ", round(m1[1,1]/float(np.sum(m1[1,:])),4)
    print "precision: ", round(m1[1,1]/float(np.sum(m1[:,1])),4)

    with open('./chooseFeature.pkl','wb') as pickle_cf:
        pickle.dump(clf,pickle_cf)


