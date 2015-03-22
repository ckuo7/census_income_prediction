__author__ = 'stevenchu'
import numpy as np
import pandas as pd
from collections import Counter


def dataBalanceBoot(y):
    # input y label
    # return balanced set of indices

    # first grab all indices of positive/negative classes
    positive_indexes = [i for i, item in enumerate(y) if item == 1]
    negative_indexes = [i for i, item in enumerate(y) if item == 0]

    # define relative sizes of clases to compute difference later
    pos_size = len(positive_indexes)
    neg_size = len(negative_indexes)

    # seeing which class is greater
    # then bootstrapping until the difference is met
    # then combining smaller class with newly bootstrapped indices
    if pos_size > neg_size:
        diff = pos_size - neg_size
        bootstrapped_neg_ind = np.random.choice(negative_indexes, diff, replace=True)
        negative_indexes.append(bootstrapped_neg_ind)
    else:
        diff = neg_size - pos_size
        bootstrapped_pos_ind = np.random.choice(positive_indexes, diff, replace=True)
        positive_indexes = np.append(positive_indexes, bootstrapped_pos_ind)

    # bringing the two indices of classes together
    # and returning that new numpy array/list
    new_train_index = np.append(negative_indexes, positive_indexes)

    return new_train_index


'''
data = pd.read_csv('census-income.data', header=None)
data.columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR', 'TARGET']

y = data['TARGET']

newy = []
for label in y:
    if label == ' 50000+.': newy.append(1)
    else: newy.append(0)

print Counter(newy)

del data['TARGET']
print data.shape

indices = dataBalanceBoot(newy)
print len(indices)
'''