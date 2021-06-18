# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
====================================
Plot ROC Curves by Sensitive Feature
====================================
"""
# %%
# # Plot ROC Curve by Sensitive Feature

# To understand the performance of a classification model, it is common to plot # the Receiver Operator Curve (ROC). The ROC Curve helps to visualize the trade # off between between sensitivity (or True Positive Rate) and specificity (1 – # False Positive Rate) for all possible thresholds of a classification model.
#
# For most classifiers, the default threshold for the decision function is .50, # or 50%. In other words, a result with a probability greater than .50 will be # classified as the positive class. A result with a probability less than .50, # will be classified as the negative class. Misclassification can result in
# false negatives and false positives.
#
# The ROC curve plots the True Postive Rate against the False Positive Rate
# for each possible threshold.  The (0,1) point indicates perfect
# classification, representing 100% sensitivity (no false negatives) and 100%
# specificity (no false positives). A random guess would give a point along a
# diagonal line (the so-called line of no-discrimination).
#
# The Area under the Curve, or AUC score gives a sense of the overall
# performance of the model. The greater the AUC score the better the model's
# ability to correctly classify the target.
#
# In visualizing the sensitive features by subgroup, a model that resulted inthe
# same ROC Curve for all subgroups could be considered a perfectly "fair"
# model. (In the real world, of course, it is not that simple and other factors
# come into play.) It was able to determine the class for each subgroup at each
# threshold equally well. If there is variation between the ROC curves, it is
# an indication that the model performs differently for different sensiive
# feature subgroups.
#
# One way to tune the model, is to adjust the threshold used by the decision
# function. In this way, we can make the model more or less sensitive. This may # result in a model that is less accurate overall, but performs more "fairly."
#
# The RocAuc module in Fairlearn allows you to easily plot ROC Curves and
# compute AUC scores by sensitive feature. Below is an example of how to use
# the module and customize the resulting plots.
#
# **Note: To use this module, you will need to install Matplotlib which is not installed as part of Fairlearn**

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

from fairlearn import datasets
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import RocAuc

# %%
## Load data

# For this example we use the UCI adult dataset where the objective is to
# predict whether a person makes more (label 1) or less (0) than $50,000 a year.

data = datasets.fetch_adult(as_frame=True)['data']
target = datasets.fetch_adult(as_frame=True)['target']

# Preview Data
print(data.shape)
print("Number of records:", len(target))
print(data.head(3))
print("Target values:", set(target))
print(target[0:3])

# %%
## Example Model
# ToDo - Add explanation of purpose here.

# Extract into X, y and A
sensitive_features = ['sex', 'race']
data_cols = [x for x in data if x not in sensitive_features]

# Predictors
ohe = OneHotEncoder()
X = ohe.fit_transform(data[data_cols])

le = LabelEncoder()
y = le.fit_transform(target)
A = data[sensitive_features]

# Split into test and train, making sure we also split the sensitive features to plot
X_train, X_test, y_train, y_test, A_train, A_test =     train_test_split(X, y, A, test_size=0.33, random_state=123)

# Fit model
clf = LogisticRegression(solver='liblinear', fit_intercept=True)
clf.fit(X_train, y_train)
y_score = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

# Get probabilies for positive prediction
y_score = y_score[:, 1]
# %%
## Default plotting configuration
# Here is a the most basic example.

# Plot ROC curves by group
rc = RocAuc(y_test, y_score, A_test)
roc_plot = rc.plot_by_group()

# %%
# However this might be difficult to read. Particularly if the data you are
# working with has a number of sensitive features to compare. You can customize
# the plot for readability by either passing fewer sensitive features to be
# plotted or by customizing the plot itself.

# %%
## Configured with Parameters
# For ease of interpretation, you can configure the size of the plot, or plot
# only selected groupings of sensitive features. It is common to plot the

# Establish plot figure
plt.figure(figsize=(8, 6))
ax = plt.gca()

rc = RocAuc(y_test, y_score, A_test)
by_group = rc.by_group()
female_only = [x for x in by_group.index if 'Female' in x]
rc.plot_by_group(
    sensitive_index=female_only,
    title = "Logistic_Regression",
    ax=ax,
    include_overall=True,
    include_baseline=True,
)
plt.show()
# %%
# It is also possible to generate the Area Under the Curve (AUC) score by sensitive feature.

auc_scores = rc.auc_by_group()
print(auc_scores)

auc_scores_female = rc.auc_by_group(female_only)
print(auc_scores_female)

# %%
## Custom Plot Style
# To customize the plot, first create a plot figure and axes using the
# Matplotlib library.

# Establish plot figure
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Plot ROC curves by group
rc = RocAuc(y_test, y_score, A_test)
rc.plot_by_group(sensitive_index=female_only, ax=ax)
plt.style.use('ggplot')
plt.show()
