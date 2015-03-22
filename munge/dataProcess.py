__author__ = 'chi-liangkuo'


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def dataProcess(df, catList, numList):

    ##############################################################
    ### df is a pandas DataFrame
    ### catList is a list of column names of categorical variables
    ### numList is a list of numerical variables
    ##############################################################

    ##############################################################
    ### Standardize the continuous variable
    ### Transform the categorical variable into one-Hot representation
    ##############################################################

    #  initialize n by 1 matrix, n is number of row of data frame
    x_ = np.zeros(df.shape[0])[:,]

    for i in catList:

        le = LabelEncoder()
        enc = OneHotEncoder()
        # append the column on the right hand side of x_
        x_ = np.c_[x_,enc.fit_transform(le.fit_transform(df[i].values)[:,np.newaxis]).toarray()]

    for i in numList:
        # standardize the variable
        sc = StandardScaler()
        x_ = np.c_[x_,sc.fit_transform(np.float_(df[i].values))]

    # get rid of the first row
    return x_[:,1:]


def dataProcess2(df):

    ### might need to uncomment the comment. not sure if the
    ### function works

    ##############################################################
    ### df is a pandas DataFrame
    ### catList is a list of column names of categorical variables
    ### numList is a list of numerical variables
    ##############################################################

    ##############################################################
    ### Standardize the continuous variable
    ### Transform the categorical variable into one-Hot representation
    ##############################################################


    df2 = pd.DataFrame()
    for i in df.columns:

        if df[i].dtype.name == 'object':
            le = LabelEncoder()
            #x = le.fit_transform(df[i].values
            #x_ = np.c_[x_,le.fit_transform(df[i].values)[:,np.newaxis]]
            df2[i] = le.fit_transform(df[i].values)
        else:
            df2[i] = df[i].values
            #x_ = np.c_[x_,df[i].values[:,np.newaxis]]

    df3 = pd.DataFrame()
    for i in df.columns:

        sc = StandardScaler()
        df3[i] = sc.fit_transform(np.float_(df2[i].values))

    return df3



