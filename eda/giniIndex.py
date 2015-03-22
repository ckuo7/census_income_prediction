__author__ = 'chi-liangkuo'
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from model.chooseFeature import chooseFeature
from munge.bootstrap import dataBalanceBoot
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pandas as pd

cols = ["age","class of worker","detailed industry recode","detailed occupation recode","education","wage per hour","enroll in edu inst last wk","marital stat",
            "major industry code","major occupation code","race","hispanic origin","sex","member of a labor union","reason for unemployment","full or part time employment stat",
            "capital gains","capital losses","dividends from stocks","tax filer stat","region of previous residence","state of previous residence","detailed household and family stat",
            "detailed household summary in household","instance weight","migration code-change in msa","migration code-change in reg","migration code-move within reg","live in this house 1 year ago","migration prev res in sunbelt",
            "num persons worked for employer","family members under 18","country of birth father","country of birth mother","country of birth self","citizenship","own business or self employed",
            "fill inc questionnaire for veteran's admin","veterans benefits","weeks worked in year","year","income level"]

def giniIndex():

    ###################################################
    ### Gini Impurity for feature selections
    ###################################################

    df = pd.read_csv('../data/census-income.data',header=None)

    y = df[41].values
    le = LabelEncoder()
    new_index = dataBalanceBoot(le.fit_transform(y))

    newdf = df.ix[new_index,:]

    clf = chooseFeature()
    impurity = clf.ftr_seln(newdf,newdf[41].values)

    count = 0
    for k,v in  Counter(impurity).items():
        print k," & ",cols[count]," & ",round(v,4)
        count +=1

def giniIndex2():

    ###################################################
    ### Gini Impurity for splitting
    ###################################################

    df = pd.read_csv('../data/census-income.data',header=None)

    y = df[41].values
    le = LabelEncoder()
    new_index = dataBalanceBoot(le.fit_transform(y))

    newdf = df.ix[new_index,:]

    clf = chooseFeature()
    impurity = clf.fit(newdf,newdf[41].values)

    for k,v in  Counter(impurity).items():
        print k,v


if __name__ == "__main__":

    ###################################################
    ### Gini Impurity for first level
    ###################################################

    giniIndex()