# %%
# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
================================
Plotting Metrics with Error Bars
================================
"""
# %%
# Load and preprocess the data set
# --------------------------------
# We download the data set using `fetch_adult` function in `fairlearn.datasets`.
# We start by importing the various modules we're going to use:
#

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from fairlearn.metrics import MetricFrame, ErrorPlotter


data = fetch_openml(data_id=1590, as_frame=True)
X = pd.get_dummies(data.data)
y_true = (data.target == '>50K') * 1
sex = data.data['sex']

# %%
# Error Metrics
# -------------
# We have many different choices for error metrics. In this notebook we'll just be using a
# Normal approximation interval (symmetric) and a Wilson score interval (asymmetric).
#
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
# https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval
#
#

z_score = 1.959964
digits = 4


def recall_normal_err(y_t, y_p):
    assert len(y_t) == len(y_p)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    error = compute_error_metric(tp/(tp+fn), tp + fn, z_score=z_score)
    return error


def wilson(p, n, digits=digits, z=z_score):
    """ Returns lower and upper bound """
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)))/np.sqrt(n)
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (round(lower_bound, digits), round(upper_bound, digits))


def compute_error_metric(metric_value, sample_size, z_score):
    return z_score*np.sqrt(metric_value*(1.0-metric_value))/np.sqrt(sample_size)


def recall_wilson_lower_bound(y_t, y_p):
    assert len(y_t) == len(y_p)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits, z_score)
    return bounds[0]


def recall_wilson_upper_bound(y_t, y_p):
    assert len(y_t) == len(y_p)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits, z_score)
    return bounds[1]


# %%
# Classifier
# ----------
# Now we use a Decision Tree to make predictions
#

classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X, y_true)
y_pred = classifier.predict(X)

# %%
# MetricFrame
# -----------
# Now we create a MetricFrame to generate the Recall wilson bounds and Recall normal error
#

# Analyze metrics using MetricFrame
metrics = {
    'Recall': recall_score,
    'Recall lower bound': recall_wilson_lower_bound,
    'Recall upper bound': recall_wilson_upper_bound,
    'Recall Error': recall_normal_err
}
metric_frame = MetricFrame(metrics, y_true, y_pred, sensitive_features=sex)

# %%
# Error Plotting
# --------------
# Finally we use the ErrorPlotter class to plot the Recall metric with its error bounds
#
# 1. We plot with the symmetric Normal Error Interval
error_mapping = {
    "Recall": {
        "symmetric_error": "Recall Error",
    }
}

ep = ErrorPlotter(metric_frame, error_mapping)
ep.plot_with_error("bar", metric="Recall",
                   title="(Symmetric) Normal Error Intervals", capsize=10, colormap="Pastel1")

# %%
# 2. We plot wiuth the asymmetric Wilson Bounds
error_mapping = {
    "Recall": {
        "upper_bound": "Recall upper bound",
        "lower_bound": "Recall lower bound"
    }
}

ep = ErrorPlotter(metric_frame, error_mapping)
ep.plot_with_error("bar", metric="Recall", title="(Asymmetric) Wilson Bounds",
                   capsize=10, colormap="Pastel2")

# %%
# Creating Custom Error Metrics
# -----------------------------
# In this tutorial we used Normal intervals and Wilson bounds to demonstrate how to plot symmetric and asymmetric metrics, respectively.
#
# However, in order to support many different metrics, we reuse the MetricFrame functionality by writing a function
# that takes in y_true and y_pred, and returns the error metric:


def error_metric_function(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    # compute error metric
    # compute custom metric function here
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits, z_score)   # compute custom metric function here
    # returns the lower bound
    return bounds[0]


# Then when creating the ErrorPlotter we pass in the mapping of the form:
error_mapping = {
    "Recall": {  # `Recall` is the Metric Name
        "upper_bound": "Recall upper bound",  # `upper_bound` is a predefined type of error metric
        # `Recall upper bound` is the column name of the metric defined in the MetricFrame
        "lower_bound": "Recall lower bound"  # `lower_bound` is another predefined type of error metric
                                             # `Recall lower bound` is the column name of the metric defined in the MetricFrame
    }
}
# %%
