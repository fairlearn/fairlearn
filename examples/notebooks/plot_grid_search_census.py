
"""
===========================
GridSearch with Census Data
===========================
"""
# %%
# This notebook shows how to use `fairlearn` and the Fairness dashboard to generate predictors
# for the Census dataset.
# This dataset is a classification problem - given a range of data about 32,000 individuals,
# predict whether their annual income is above or below fifty thousand dollars per year.
#
# For the purposes of this notebook, we shall treat this as a loan decision problem.
# We will pretend that the label indicates whether or not each individual repaid a loan in
# the past.
# We will use the data to train a predictor to predict whether previously unseen individuals
# will repay a loan or not.
# The assumption is that the model predictions are used to decide whether an individual
# should be offered a loan.
#
# We will first train a fairness-unaware predictor and show that it leads to unfair
# decisions under a specific notion of fairness called *demographic parity*.
# We then mitigate unfairness by applying the `GridSearch` algorithm from `fairlearn` package.

# %%
# Load and preprocess the data set
# --------------------------------
# For simplicity, we import the data set from the `shap` package, which contains the data in
# a cleaned format.
# We start by importing the various modules we're going to use:

from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate

from sklearn import svm, neighbors, tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import shap

import numpy as np

# %%
# We can now load and inspect the data from the `shap` package:

X_raw, Y = shap.datasets.adult()
X_raw

# %%
# We are going to treat the sex of each individual as a protected
# attribute (where 0 indicates female and 1 indicates male), and in
# this particular case we are going separate this attribute out and drop it
# from the main data.
# We then perform some standard data preprocessing steps to convert the
# data into a format suitable for the ML algorithms

A = X_raw["Sex"]
X = X_raw.drop(labels=['Sex'],axis = 1)
X = pd.get_dummies(X)

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

le = LabelEncoder()
Y = le.fit_transform(Y)

# %%
# Finally, we split the data into training and test sets:

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled, 
                                                    Y, 
                                                    A,
                                                    test_size = 0.2,
                                                    random_state=0,
                                                    stratify=Y)

# Work around indexing bug
X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# Improve labels
A_test = A_test.map({ 0:"female", 1:"male"})