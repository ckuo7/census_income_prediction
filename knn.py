__author__ = 'chi-liangkuo'


from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import LabelEncoder
from dataProcess import dataProcess
from dataBalance import dataBalance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit

def knn_eval():

    np.random.seed(10)
    df = pd.read_csv('./census-income.data',header=None)

    catList = [1,2,3,4,8,9,12,19,23,32]
    numList = [0,16,18,30,39]

    X_ = dataProcess(df,catList,numList)
    le = LabelEncoder()
    y_ = le.fit_transform(df[41].values)

    print "X: \n",X_," x.shape: ",X_.shape
    print "y: \n",y_

    new_train_index = dataBalance(y_,0.05)

    # 10% of training set
    # eua [0.8232, 0.857, 0.8593, 0.8591, 0.8589, 0.8586, 0.8597, 0.8602, 0.8608, 0.8614, 0.8604]
    # eda [0.8232, 0.8534, 0.8559, 0.8568, 0.8572, 0.8584, 0.8589, 0.8595, 0.8606, 0.8608, 0.8593]
    # jua [0.7854, 0.8268, 0.8353, 0.8291, 0.8358, 0.8314, 0.8366, 0.8358, 0.8364, 0.8395, 0.838]
    # euf [0.8248, 0.8614, 0.864, 0.8604, 0.8639, 0.861, 0.8654, 0.866, 0.8659, 0.8689, 0.8687]
    # edf [0.8248, 0.8576, 0.8604, 0.8612, 0.862, 0.8632, 0.8644, 0.8651, 0.8666, 0.8685, 0.8677]
    # juf [0.7747, 0.8242, 0.8343, 0.8229, 0.8352, 0.8274, 0.8377, 0.8376, 0.8374, 0.8442, 0.844]

    X = X_[new_train_index,:]
    y = y_[new_train_index]


    K = [1,5,7,8,9,10,13,15,20,50,70]

    eua = []
    euf = []
    eda = []
    edf = []
    jua = []
    juf = []
    for i in K:

        print "number of neighbor: ",i

        a1 = cross_val_score(KNeighborsClassifier(n_neighbors=i),X,y,cv=5,scoring="accuracy")
        print "Euclidean distance with uniform weight..."
        print "accuracy: ", np.mean(a1)
        eua.append(round(np.mean(a1),4))
        f1 = cross_val_score(KNeighborsClassifier(n_neighbors=i),X,y,cv=5,scoring="f1")
        print "f1: ",np.mean(f1),"\n"
        euf.append(round(np.mean(f1),4))


        print "Euclidean distance with distance weight..."
        a2 = cross_val_score(KNeighborsClassifier(n_neighbors=i,weights='distance'),X,y,cv=5,scoring="accuracy")
        print "accuracy: ",np.mean(a2)
        eda.append(round(np.mean(a2),4))
        f2 = cross_val_score(KNeighborsClassifier(n_neighbors=i,weights='distance'),X,y,cv=5,scoring="f1")
        print "f1: ",np.mean(f2),"\n"
        edf.append(round(np.mean(f2),4))

        print "Jacaard distance with uniform weight..."
        a3 = cross_val_score(KNeighborsClassifier(n_neighbors=i,metric='jaccard'),X,y,cv=5,scoring="accuracy")
        print "accuracy: ",np.mean(a3)
        jua.append(round(np.mean(a3),4))
        f3 = cross_val_score(KNeighborsClassifier(n_neighbors=i,metric='jaccard'),X,y,cv=5,scoring="f1")
        print "f1: ",np.mean(f3),"\n"
        juf.append(round(np.mean(f3),4))



    L1, = plt.plot(K, eua, color='b', lw=1.5, label='Euclidean distance\nuniform weight')
    L2, = plt.plot(K, eda, color='green', lw=1.5, label='Euclidean distance\ndistance weight')
    L3, = plt.plot(K, jua, color='red', lw=1.5, label='Jaccard distance\nuniform weight')
    plt.title('Accuracy vs Number of Neighbor', fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Number of Neighbors', fontsize=16)
    plt.subplots_adjust(right=0.75)
    plt.legend( handles = [L1,L2,L3],loc=1,bbox_to_anchor=(1.375,0.6),fontsize = 10  )
    plt.show()


    L4, = plt.plot(K, euf, color='b', lw=1.5, label='Euclidean distance\nuniform weight')
    L5, = plt.plot(K, edf, color='green', lw=1.5, label='Euclidean distance\ndistance weight')
    L6, = plt.plot(K, juf, color='red', lw=1.5, label='Jaccard distance\nuniform weight')
    plt.title('F1 vs Number of Neighbor', fontsize=20)
    plt.ylabel('F1', fontsize=16)
    plt.xlabel('Number of Neighbors', fontsize=16)
    plt.subplots_adjust(right=0.75)
    plt.legend( handles = [L4,L5,L6],loc=1,bbox_to_anchor=(1.375,0.6),fontsize = 10  )
    plt.show()

    print "eua", eua
    print "eda", eda
    print "jua", jua
    print "euf", euf
    print "edf", edf
    print "juf", juf


if __name__ == "__main__":

    start = timeit.default_timer()
    knn_eval()
    stop = timeit.default_timer()
    print "run time: ",(stop-start)