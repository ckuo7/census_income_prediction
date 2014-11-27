__author__ = 'chi-liangkuo'


from chooseFeature import chooseFeature
from bootstrap import dataBalanceBoot
from sklearn.preprocessing import LabelEncoder
from collections import Counter

import pandas as pd

def giniIndex():

    ###################################################
    ### Gini Impurity for feature selections
    ###################################################

    df = pd.read_csv('./census-income.data',header=None)

    y = df[41].values
    le = LabelEncoder()
    new_index = dataBalanceBoot(le.fit_transform(y))

    newdf = df.ix[new_index,:]

    clf = chooseFeature()
    impurity = clf.ftr_seln(newdf,newdf[41].values)

    for k,v in  Counter(impurity).items():
        print k,v

def giniIndex2():

    ###################################################
    ### Gini Impurity for splitting
    ###################################################

    df = pd.read_csv('./census-income.data',header=None)

    y = df[41].values
    le = LabelEncoder()
    new_index = dataBalanceBoot(le.fit_transform(y))

    newdf = df.ix[new_index,:]

    clf = chooseFeature()
    impurity = clf.fit(newdf,newdf[41].values)

    for k,v in  Counter(impurity).items():
        print k,v


if __name__ == "__main__":
    giniIndex2()