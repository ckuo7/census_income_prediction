__author__ = 'chi-liangkuo'




from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from dataProcess import dataProcess
from dataBalance import dataBalance
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

def svc_eval():


    np.random.seed(10)

    df = pd.read_csv('./census-income.data',header=None)

    catList = [1,2,3,4,8,9,12,19,23,32]
    numList = [0,16,18,30,39]

    X_ = dataProcess(df,catList,numList)

    le = LabelEncoder()
    y_ = le.fit_transform(df[41].values)

    print "X: \n",X_," x.shape: ",X_.shape
    print "y: \n",y_

    new_train_index = dataBalance(y_,0.025)



    X = X_[new_train_index,:]
    y = y_[new_train_index]


    gamma = [0,0.01,0.1,1]
    C = [0.1,0.3,1,3,10,30]


    a_dic = {}
    f_dic = {}
    for g in gamma:

        A = []
        F = []
        for c in C:
            print "gamma = ",g,", C = ",c

            a = cross_val_score(SVC(gamma=g,C=c),X,y,scoring="accuracy",cv=5)
            a_ = np.mean(a)
            print "5 fold accuracy = ",round(a_,4)
            f = cross_val_score(SVC(gamma=g,C=c),X,y,scoring="f1",cv=5)
            f_ = np.mean(f)
            print "5 fold f1 score = ",round(f_,4)
            A.append(round(a_,4))
            F.append(round(f_,4))
        a_dic[g] = A
        f_dic[g] = F

    for k,v in a_dic.items():
        print k,v
        print f_dic[k]
    #print a_dic[0]
    L1, = plt.plot(C, a_dic[0], color='blue', lw=1.5, label='Gamma = 0.0')
    L2, = plt.plot(C, a_dic[0.01], color='green', lw=1.5, label='Gamma = 0.01')
    L3, = plt.plot(C, a_dic[0.1], color='red', lw=1.5, label='Gamma = 0.1')
    L4, = plt.plot(C, a_dic[1], color='black', lw=1.5, label='Gamma = 1')

    plt.title('Accuracy vs C', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('C', fontsize=16)
    plt.subplots_adjust(right=0.75)
    plt.legend( handles = [L1,L2,L3,L4],loc=1,bbox_to_anchor=(1.375,0.6),fontsize = 10  )
    plt.show()

    L1, = plt.plot(C, f_dic[0], color='blue', lw=1.5, label='Gamma = 0.0')
    L2, = plt.plot(C, f_dic[0.01], color='green', lw=1.5, label='Gamma = 0.01')
    L3, = plt.plot(C, f_dic[0.1], color='red', lw=1.5, label='Gamma = 0.1')
    L4, = plt.plot(C, f_dic[1], color='black', lw=1.5, label='Gamma = 1')

    plt.title('F1-score vs C', fontsize=20)
    plt.ylabel('F1-score', fontsize=16)
    plt.xlabel('C', fontsize=16)
    plt.subplots_adjust(right=0.75)
    plt.legend( handles = [L1,L2,L3,L4],loc=1,bbox_to_anchor=(1.375,0.6),fontsize = 10  )
    plt.show()


if __name__ == "__main__":

    start = timeit.default_timer()
    svc_eval()
    stop = timeit.default_timer()
    print "run time: ",(stop-start)