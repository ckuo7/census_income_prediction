# Census Income Prediction: The Problem

This data consists of a row per person. There are 40 features associated with each person and the goal is to classify each person as someone who does or does not make over $50,000 per year. Features consist of both numerical (such as age) as well as categorical (such as race) variables.

# Loading the Data

After unpacking the tar file, you should see three distinct files:

 1. census-income.data - this file contains the training data.
 
 2. census-income.test - this file contains the testing data.
 
 3. target.txt - this text file contains information about the training data (how many distinct values there are for each feature).

# EDA

The first thing we did was investigate the general shape of our data. We plotted histograms as well as kept track of the number of distinct levels of each feature. Under the "eda" file, you will find the following three scripts we used to help us do so:

 1. brainstorm.py is a Python script that iterates through each column in our training, generating bar plots for the distinct values each feature takes. This is more informative for categorical as opposed to numerical (continuous) features.

 2. featureReport.py is a Python script that similarly iterates through each column in our training data, generating percentages of the distinct values for each feature. For example, this script will tell you that we have 47.88% of people in our training data are males, while 52.12% of people in our training data are females. Again, this is more informative for categorical rather than numerical features.

 3. giniIndex.py is a Python script that iterates through each column in our training data, generating the gini coefficient of each. This is useful for telling us how "pure" each feature is.

# Data Munging

Because the dataset is unbalanced (more people earn less than $50,000 per year than more), we wanted a way of boostrapping our data to acheive 

 1. bootstrap.py
 
 2. dataBalance.py - 
 
 3. dataBalanceRatio.py
 
 4. dataProcess.py
 
 5. dataProcessLabel.py
 
 6. dataProcessOne.py

# Running the Models

# Testing









# Summary

We applied a number of various machine learning algorithms to this problem. Along the way, we tuned the hyper-parameters of our algorithms. While investigating not only these algorithms and their associated hyper-parameters, we dealt with missing as well as unbalanced data. Using F1 scores as our metric for success, we ultimately achieved the highest F1 score of 0.5911 with an SVM classifier with an RBF kernel.

# Introduction

# The Data

# Data Cleaning

# Model Selection

# Final Results

# Discussion

# Conclusion
