__author__ = 'chi-liangkuo'


from math import floor
import numpy as np

def dataBalance(y,percent):

    # input y label
    # return the training set index

    training_size =  floor(len(y)*percent)


    positive_indexes = [i for i, item in enumerate(y) if item==1]
    negative_indexes = [i for i, item in enumerate(y) if item==0]

    # if training size less than the number of negative training examples,
    # then don't resample the negative example


    if training_size <= 187141:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=True)
        neg_train_index = np.random.choice(negative_indexes,training_size,replace=False)
    else:
        pos_train_index = np.random.choice(positive_indexes,training_size,replace=True)
        neg_train_index = np.random.choice(negative_indexes,training_size,replace=True)

    new_train_index = np.append(pos_train_index,neg_train_index)
    np.random.shuffle(new_train_index)

    #shuffle again!!

    return new_train_index
