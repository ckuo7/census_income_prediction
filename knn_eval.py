__author__ = 'chi-liangkuo'


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from dataProcess import dataProcess
from dataBalance import dataBalance
from bootstrap import dataBalanceBoot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

def cosine_similarity(label1, label2):

    return np.dot(label1,label2)/(np.sqrt(label1.dot(label1))*np.sqrt(label2).dot(label2))
def knn_eval(X,y):

    clf = KNeighborsClassifier()

    K = [1,3,4,5,6,7,10,15,20]

    A = []
    B = []
    C = []
    P = []
    R = []
    RA = []
    for i in K:

        a = cross_val_score(KNeighborsClassifier(n_neighbors=i),X,y,cv=3,scoring="accuracy")
        b = cross_val_score(KNeighborsClassifier(n_neighbors=i,weights='distance'),X,y,cv=3,scoring="accuracy")
        c = cross_val_score(KNeighborsClassifier(n_neighbors=i,metric='jaccard'),X,y,cv=3,scoring="accuracy")
        # p = cross_val_score(clf,X,y,cv=5,scoring="precision")
        # r = cross_val_score(clf,X,y,cv=5,scoring="recall")
        # r2 = cross_val_score(clf,X,y,cv=5,scoring="roc_auc")

        A.append(np.mean(a))
        B.append(np.mean(b))
        C.append(np.mean(c))
        # P.append(np.mean(p))
        # R.append(np.mean(r))
        # RA.append(np.mean(r2))
    print "A: ",A
    print "B: ",B
    L1, = plt.plot(K, A, color='b', lw=1.5, label='Euclidean distance\nuniform weight')
    L2, = plt.plot(K, B, color='green', lw=1.5, label='Euclidean distance\ndistance weight')
    L3, = plt.plot(K, C, color='red', lw=1.5, label='Jaccard distance\nuniform weight')
    plt.title('Accuracy vs Number of Neighbor', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Number of Neighbors', fontsize=16)
    plt.subplots_adjust(right=0.75)
    plt.legend( handles = [L1,L2,L3],loc=1,bbox_to_anchor=(1.375,0.6),fontsize = 10  )

    plt.show()

if __name__ == "__main__":

    start = timeit.default_timer()

    df = pd.read_csv('./census-income.data',header=None)
    l = [0,4,30,31,39]
    X_ = dataProcess(df[l])
    le = LabelEncoder()
    y_ = le.fit_transform(df[41].values)

    new_train_index = dataBalance(y_,0.05)
    #new_train_index = dataBalanceBoot(y_)

    X = X_[new_train_index,:]
    y = y_[new_train_index]
    print X_.shape
    print y

    knn_eval(X,y)


    stop = timeit.default_timer()
    print "run time: ",(stop-start)