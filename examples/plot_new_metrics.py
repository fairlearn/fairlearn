# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
==============================
Metrics with Multiple Features
==============================
"""
# %%
# This notebook demonstrates the new API for metrics, which supports
# multiple sensitive and conditional features. This example does not
# contain a proper discussion of how fairness relates to the dataset
# used, although it does highlight issues which users may want to
# consider when analysing their datasets.
#
# We are going to consider a lending scenario, supposing that we have
# a model which predicts whether or not a particular customer will
# repay a loan. This could be used as the basis of deciding whether
# or not to offer that customer a loan. With traditional metrics,
# we would assess the model using:
#
# - The 'true' values from the test set
# - The model predictions from the test set
#
# Our fairness metrics compute group-based fairness statistics.
# To use these, we also need categorical columns from the test
# set. For this example, we will include:
#
# - The sex of each individual (two unique values)
# - The race of each individual (three unique values)
# - The credit score band of each individual (three unique values)
# - Whether the loan is considered 'large' or 'small'
#
# An individual's sex and race should not affect a lending decision,
# but it would be legitimate to consider an individual's credit score
# and the relative size of the loan which they desired.
#
# A real scenario will be more complicated, but this will serve to
# illustrate the use of the new metrics.
#
# Getting the Data
# ================
#
# *This section may be skipped. It simply creates a dataset for
# illustrative purposes*
#
# We will use the well-known UCI 'Adult' dataset as the basis of this
# demonstration. This is not for a lending scenario, but we will regard
# it as one for the purposes of this example. We will use the existing
# 'race' and 'sex' columns (trimming the former to three unique values),
# and manufacture credit score bands and loan sizes from other columns.
# We start with some uncontroversial `import` statements:

import functools
import sklearn.metrics as skm
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.metrics import GroupedMetric, make_derived_metric
from fairlearn.metrics import accuracy_score_group_min, accuracy_score_difference


# %%
# Next, we import the data:

data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
Y = (data.target == '>50K') * 1

# %%
# For purposes of clarity, we consolidate the 'race' column to have
# three unique values:


def race_transform(input_str):
    """Reduce values to White, Black and Other."""
    result = 'Other'
    if input_str == 'White' or input_str == 'Black':
        result = input_str
    return result


X_raw['race'] = X_raw['race'].map(race_transform).fillna('Other')
print(np.unique(X_raw['race']))

# %%
# Now, we manufacture the columns for the credit score band and
# requested loan size. These are wholly constructed, and not
# part of the actual dataset in any way. They are simply for
# illustrative purposes.


def marriage_transform(m_s_string):
    """Perform some simple manipulations."""
    result = 'Low'
    if m_s_string.startswith("Married"):
        result = 'Medium'
    elif m_s_string.startswith("Widowed"):
        result = 'High'
    return result


def occupation_transform(occ_string):
    """Perform some simple manipulations."""
    result = 'Small'
    if occ_string.startswith("Machine"):
        result = 'Large'
    return result


col_credit = X_raw['marital-status'].map(marriage_transform).fillna('Low')
col_credit.name = "Credit Score"
col_loan_size = X_raw['occupation'].map(occupation_transform).fillna('Small')
col_loan_size.name = "Loan Size"

A = X_raw[['race', 'sex']]
A['Credit Score'] = col_credit
A['Loan Size'] = col_loan_size
A

# %%
# With the data imported, we perform some standard processing, and a test/train split:
le = LabelEncoder()
Y = le.fit_transform(Y)

le = LabelEncoder()

sc = StandardScaler()
X_dummies = pd.get_dummies(X_raw)
X_scaled = sc.fit_transform(X_dummies)
X_scaled = pd.DataFrame(X_scaled, columns=X_dummies.columns)

X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled, Y, A,
                                                                     test_size=0.3,
                                                                     random_state=12345,
                                                                     stratify=Y)

# Ensure indices are aligned
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# %%
# Finally, we train a simple model on the data, and generate
# some predictions:


unmitigated_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)
unmitigated_predictor.fit(X_train, Y_train)

Y_pred = unmitigated_predictor.predict(X_test)

# %%
# Analysing the Model with Metrics
# ================================
#
# After our data manipulations and model training, we have the following
# from our test set:
#
# - A vector of true values called ``Y_test``
# - A vector of model predictions called ``Y_pred``
# - A DataFrame of categorical features relevant to fairness called ```A_test``
#
# In a traditional model analysis, we would now look at some metrics
# evaluated on the entire dataset. Suppose in this case, the relevant
# metrics are :func:`sklearn.metrics.precision_score()` and
# :func:`sklearn.metrics.fbeta_score` (with 
# ``beta=0.6``).
# We can evaluate these metrics using SciKit-Learn:

print("Precision:", skm.precision_score(Y_test, Y_pred))
print("fbeta:", skm.fbeta_score(Y_test, Y_pred, beta=0.6))

# %%
# We know that there are sensitive features in our data, and we want to
# ensure that we're not harming individuals due to membership in any of
# these groups. For this purpose, Fairlearn provides the
# :class:`fairlearn.metrics.GroupedMetric`
# class. Let us construct an instance of this class, and then look at
# its capabilities:

fbeta_06 = functools.partial(skm.fbeta_score, beta=0.6)

metric_fns = {'precision': skm.precision_score, 'fbeta_06': fbeta_06}

grouped_on_sex = GroupedMetric(metric_fns,
                               Y_test, Y_pred,
                               sensitive_features=A_test['sex'])

# %%
# The :class:`fairlearn.metrics.GroupedMetric` object requires a
# minimum of four arguments:
#
# 1. The underlying metric function(s) to be evaluated
# 2. The true values
# 3. The predicted values
# 4. The sensitive feature values
#
# These are all passed as arguments to the constructor. If more than
# one underlying metric is required (as in this case), then we must
# provide them in a dictionary.
#
# The underlying metrics must have a signature ``fn(y_true, y_pred)``,
# so we have to use :func:`functools.partial` on ``fbeta_score()`` to
# furnish ``beta=0.6`` (we will show how to pass in extra array
# arguments such as sample weights shortly).
#
# We will now take a closer look at the :class:`fairlearn.metrics.GroupedMetric`
# object. First, there is the ``overall`` property, which contains
# the metrics evaluated on the entire dataset. We see that this contains the
# same values calculated above:

assert grouped_on_sex.overall['precision'] == skm.precision_score(Y_test, Y_pred)
assert grouped_on_sex.overall['fbeta_06'] == skm.fbeta_score(Y_test, Y_pred, beta=0.6)
print(grouped_on_sex.overall)

# %%
# The other property in the :class:`fairlearn.metrics.GroupedMetric` object
# is ``by_group``. This contains the metrics evaluated on each subgroup defined
# by the categories in the ``sensitive_features=`` argument. In this case, we
# have results for males and females:

print(grouped_on_sex.by_group)