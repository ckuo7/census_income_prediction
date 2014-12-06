

import sklearn
import numpy as np
import pandas as pd
import collections
from sklearn.cross_validation import cross_val_score

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from dataBalance import dataBalance

class chooseFeature(sklearn.base.BaseEstimator):
    """
    This defines a classifier that predicts on the basis of
      the feature that was found to have the best weighted purity, based on splitting all
      features according to their mean value. Then, for that feature split, it predicts
      a new example based on the mean value of the chosen feature, and the majority class for
      that split.
      You can of course define more variables!
    """

    def __init__(self):
        # if we haven't been trained, always return 1
        self.classForGreater= 1
        self.classForLeq = 1
        self.chosenFeature = 0
        self.type = "chooseFeatureClf"

    def impurity(self, labels):
        pass
        # ## TODO: Your code here
        # labels should be a numpy array of categorical variable
        # gini impurity = \sum_i^K f_i*(1-f_i)
        labels_len = len(labels)
        labelsCounter = collections.Counter(labels)
        impurity = 0
        for k,v in labelsCounter.items():
            f = v/float(labels_len)
            impurity += f*(1-f)
        return impurity



    def weighted_impurity(self, list_of_label_lists):
        pass
        # ## TODO: Your code here, uses impurity
        # list of label list should have [ [],[] ] structure
        set_size = np.array([len(i) for i in list_of_label_lists])
        set_total = np.sum(set_size)
        weighted = np.array([self.impurity(list_of_label_lists[i])*set_size[i]/float(set_total) for i in range(len(set_size))])
        return np.sum(weighted)

    def ftr_seln(self, data, labels):
        """return: index of feature with best weighted_impurity, when split
        according to its mean value; you are permitted to return other values as well,
        as long as the the first value is the index
        """
        # pass
        # TODO: Your code here, uses weighted_impurity

        category = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,19,20,21,22,23,25,26,
                    27,28,29,31,32,33,34,35,36,37,38,40]
        numeric = [0,5,16,17,18,24,30,39]


        impurity ={}
        df = data

        for i in category:

            labelOfList = []
            for category  in set(df[i]):

                ind = [j for j, item in enumerate(df[i].values) if item==category]
                labelOfList.append(labels[ind])


            impurity[i] = self.weighted_impurity(labelOfList)

        for n in numeric:

            labelOfList = []
            if len(set(df[n].values)) == 1:
                # if it happened that example have the same value in this feature
                impurity[n] = self.weighted_impurity([labels])
            else:
                m = df[n].mean()
                ind1 = [j for j, item in enumerate(df[n].values) if item > m]  # index above mean
                ind2 = [j for j, item in enumerate(df[n].values) if item <= m] # index below or equal to mean

                labelOfList.append(labels[ind1])
                labelOfList.append(labels[ind2])
                impurity[n] = self.weighted_impurity(labelOfList)

        return impurity


    def fit(self, data, labels):
        """
        Inputs: data: a list of X vectors
        labels: Y, a list of target values
        """
        # ## TODO: Your code here, uses ftr_seln
        # get the index with minimum gini index

        # df = pd.DataFrame(data)
        # data is a dataframe
        df = pd.DataFrame(data)
        impurity = self.ftr_seln( df, labels)

        gini = 1000000
        feature = -1
        for k,v in impurity.items():
            if v < gini:
                gini = v
                feature = k
            else:
                continue

        print "choosefeature: ",feature
        self.feature = feature

        # # Calculate Gini Indices with leave one out approach
        # impurity2 = {}
        # categories = Counter(df[feature].values)
        # for category in categories:
        #
        #     ind1 = [j for j, item in enumerate(df[k].values) if item == category]  # get the indices equal to category
        #     ind2 = [j for j, item in enumerate(df[k].values) if item != category]  # get the indices not equal to category
        #
        #     labelOfList = []
        #     labelOfList.append(labels[ind1])
        #     labelOfList.append(labels[ind2])
        #     impurity2[category] = self.weighted_impurity(labelOfList)
        #
        #
        # return impurity2


        # calculate the mean of selected feature
        self.mean = df[self.feature].mean()

        # if there is only one category in that features
        if len(set(df[self.feature].values)) == 1:
            self.case = True
            # Return the most common one
            self.equal_mean = Counter(labels).most_common(1)[0][0]
        else:
            self.case = False
            self.above_mean = Counter(pd.Series(labels)[ df[ df[self.feature] > self.mean ].index ]).most_common(1)[0][0]
            self.below_mean = Counter(pd.Series(labels)[ df[ df[self.feature] <= self.mean ].index ]).most_common(1)[0][0]


    def predict(self, testData):
        """
        Input: testData: a list of X vectors to label.
        Check the chosen feature of each
        element of testData and make a classification decision based on it
        """
        # ## TODO: Your code here
        output = []
        if self.case:
            return np.array([self.equal_mean for i in testData])

        else:
            for j in testData:
                if j[self.feature] > self.mean:
                    output.append(self.above_mean)
                else:
                    output.append(self.below_mean)
            return np.array(output)


def eval():



    df = pd.read_csv('./census-income.data',header=None)
    y = df[41].values
    le = LabelEncoder()
    newy = le.fit_transform(y)
    new_index = dataBalance(newy,0.93)

    newdf = df.ix[new_index,:]
    clf = chooseFeature()

    # b = cross_val_score(clf,newdf,newy[new_index],cv=5,scoring="precision")
    # print "precision", np.mean(b)
    # 5 folds cross validation precision : precision 0.102141840525

    a = cross_val_score(clf,newdf,newy[new_index],cv=5,scoring="accuracy")
    print "accuracy: ", np.mean(a)

    f = cross_val_score(clf,newdf,newy[new_index],cv=5,scoring="f1")
    print "f1: ", np.mean(f)






if __name__ == '__main__':
    eval()

