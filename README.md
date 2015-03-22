# Census Income Prediction: The Problem

This data consists of a row per person. There are 40 features associated with each person and the goal is to classify each person as someone who does or does not make over $50,000 per year. Features consist of both numerical (such as age) as well as categorical (such as race) variables.

# Loading the Data

After unpacking the tar file, you should see three distinct files:

 1. census-income.data - this file contains the training data.
 
 2. census-income.test - this file contains the testing data.
 
 3. target.txt - this text file contains information about the training data (how many distinct values there are for each feature, etc).

# EDA

The first thing we did was investigate the general shape of our data. We plotted histograms as well as kept track of the number of distinct levels of each feature. Under the "eda" file, you will find the following three scripts we used to help us do so:

 1. brainstorm.py is a Python script that iterates through each column in our training, generating bar plots for the distinct values each feature takes. This is more informative for categorical as opposed to numerical (continuous) features.

 2. featureReport.py is a Python script that similarly iterates through each column in our training data, generating percentages of the distinct values for each feature. For example, this script will tell you that we have 47.88% of people in our training data are males, while 52.12% of people in our training data are females. Again, this is more informative for categorical rather than numerical features.

 3. giniIndex.py is a Python script that iterates through each column in our training data, generating the gini coefficient of each. This is useful for telling us how "pure" each feature is.

# Data Munging

Our data munging process is divided into two tasks. The first is balancing our dataset according to specified ratios. The second is transforming our data to more manageable levels. The first two files cover the first task, while the remainder cover the latter one.

Because the dataset is unbalanced (more people earn less than $50,000 per year than more), we wanted a way of boostrapping our data. Specifically, we wanted to investigate two scenarios: the first was to bootstrap to have an even number of positive and negative classes and the second was when we had specified ratio of positive to negative classes.

 1. dataBalance.py has two functions: dataBalance and dataUnbalance. The first is a way of achieving an evenly balanced dataset of both positive and negative classes, while the second is used to achieve an unbalance dataset in a specified ratio (say, 30% of the negative class and 70% of the positive). This particular file was used in the SVM, K-nn, as well as chooseFeature classifier.
 
 2. dataBalanceRatio.py has the same two functions as above. The only difference is that we used this particular file in the random forest as well as logistic classifiers.

In order to run some of sklearn's algorithms on our data, the user must transform their categorical features into indicator variables. We used sklearn's OneHotEncoding method for this. Additionally, to increase performace time, we chose to normalize our numerical features. All three of the following files have functions to do this. The user passes their dataframe into one of these functions, and outputted will be a newly transformed dataframe. We employed different files for different algorithms listed below.

 1. dataProcess.py for the SVM, K-nn, and choosefeature classifiers.
 
 2. dataProcessLabel.py for the random forest classifier.
 
 3. dataProcessOne.py for the logistic classifier.

# Running the Models

BLAH BLAH BLAH

 1. chooseFeature.py
 
 2. knn.py
 
 3. svc.py
 
 4. rf_logistic_hyperparameters.py

# Testing

BLAH BLAH BLAH







# Summary

We applied a number of various machine learning algorithms to this problem. Along the way, we tuned the hyper-parameters of our algorithms. While investigating not only these algorithms and their associated hyper-parameters, we dealt with missing as well as unbalanced data. Using F1 scores as our metric for success, we ultimately achieved the highest F1 score of 0.5911 with an SVM classifier with an RBF kernel.
