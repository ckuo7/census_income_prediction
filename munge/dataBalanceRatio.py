__author__ = 'chi-liangkuo'

from math import floor
import numpy as np

def dataBalance(y,percent):


    ##############################################################
    ###   y: target labels
    ###   percent: how many 2*percent of training examples we want
    ###            to sample
    ##############################################################

    ##############################################################
    ### bootstrape the positive example to get a balanced data set
    ### if training size less than the number of negative training examples,
    ### then don't resample the negative example
    ##############################################################


    training_size =  floor(len(y)*percent)
    positive_indexes = [i for i, item in enumerate(y) if item==1]
    negative_indexes = [i for i, item in enumerate(y) if item==0]


    if percent <= 0.05:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=False)
        neg_train_index = np.random.choice(negative_indexes,training_size,replace=False)

    elif percent <= 0.93:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=True)
        neg_train_index = np.random.choice(negative_indexes,training_size,replace=False)

    else:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=True)
        neg_train_index = np.random.choice(negative_indexes,training_size,replace=True)

    new_train_index = np.append(pos_train_index,neg_train_index)
    np.random.shuffle(new_train_index)

    print "Number of training example: ",len(new_train_index),". Number of positive example: ",len(pos_train_index),\
        ". Number of negative example: ",len(neg_train_index),"\n"

    return new_train_index

def dataUnbalance(y, percent, ratio):

    training_size = floor(len(y)*percent)

    positive_indexes = [i for i, item in enumerate(y) if item==1]
    negative_indexes = [i for i, item in enumerate(y) if item==0]


    if percent <= 0.05:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=False)
        neg_train_index = np.random.choice(negative_indexes,floor(training_size*ratio),replace=False)

    elif percent <= 0.93:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=True)
        neg_train_index = np.random.choice(negative_indexes,floor(training_size*ratio),replace=False)

    else:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=True)
        neg_train_index = np.random.choice(negative_indexes,floor(training_size*ratio),replace=True)


    new_train_index = np.append(pos_train_index,neg_train_index)
    np.random.shuffle(new_train_index)

    print "Number of training example: ",len(new_train_index),". Number of positive example: ",len(pos_train_index),\
        ". Number of negative example: ",len(neg_train_index),"\n"

    return new_train_index
