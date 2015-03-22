__author__ = 'chi-liangkuo'
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def dataProcessOne(df):

    #######################################################################
    ### input is a pandas data frame
    ### standardize the continuous variable
    ### transform the categorical variable into labels
    ### initialize n by 1 matrix, n is number of row of data frame
    #######################################################################

    x_ = np.zeros(df.shape[0])[:, ]

    for i in df.columns:
        # if it's categorical variable, then transform into oneHot rep
        if df[i].dtype.name == 'object':
            le = LabelEncoder()
            enc = OneHotEncoder()
            # append the column on the right hand side of x_
            x_ = np.c_[x_,enc.fit_transform(le.fit_transform(df[i].values)[:,np.newaxis]).toarray()]

        else:
            # standardize the variable
            sc = StandardScaler()
            x_ = np.c_[x_,sc.fit_transform(np.float_(df[i].values))]

    # get rid of the first row
    return x_[:,1:]