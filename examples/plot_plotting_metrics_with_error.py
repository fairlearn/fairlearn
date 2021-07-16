# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
================================
Plotting Metrics with Error Bars
================================
"""
# %%
# Load and preprocess the data set
# ================================
# We download the data set using `fetch_openml` function in `sklearn.datasets`.
# The original Adult data set can be found at https://archive.ics.uci.edu/ml/datasets/Adult
# There are some caveats to using this dataset, but we will use it solely as an example
# to demonstrate the plotting metrics with error bars functionality
#
# We start by importing the various modules we're going to use:

# TODO: Remove in final version (helps run with local packages)
import matplotlib.pyplot as plt
from fairlearn.experimental.enable_metric_frame_plotting import plot_metric_frame
from fairlearn.metrics import MetricFrame
from sklearn.datasets import fetch_openml
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


data = fetch_openml(data_id=1590, as_frame=True)
X = data.data
y = (data.target == '>50K') * 1
X_train, X_test, y_train_true, y_test_true = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_processed = pd.get_dummies(X_train)
X_test_processed = pd.get_dummies(X_test)
test_set_sex = X_test['sex']

# %%
# Error Metrics
# =============
# We have many different choices for error metrics. In this notebook we'll just be using a
# `Normal approximation interval <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval>` (symmetric)
# and a `Wilson score interval <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval>` (asymmetric).

# We aim to create a 95% confidence interval, so we use a `z_score` of 1.959964
z_score = 1.959964
digits_of_precision = 4


def wilson(p, n, digits=digits_of_precision, z=z_score):
    """Return lower and upper bound"""
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)))/np.sqrt(n)
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (round(lower_bound, digits), round(upper_bound, digits))


def compute_error_metric(metric_value, sample_size, z_score):
    """Compute Standard Error Calculation (for Binary Classification)

    Assumes infinitely large population,
    Should be used when the sampling fraction is small.
    For sampling fraction > 5%, may want to use finite population correction
    https://en.wikipedia.org/wiki/Margin_of_error

    Note:
        Returns absolute error (%)
    """
    return z_score*np.sqrt(metric_value*(1.0-metric_value))/np.sqrt(sample_size)


def recall_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits_of_precision, z_score)
    return bounds


def recall_normal_err(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    error = compute_error_metric(tp/(tp+fn), tp + fn, z_score=z_score)
    return (error, error)


def accuracy_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    bounds = wilson(score, len(y_true), digits_of_precision, z_score)
    return bounds


def accuracy_normal_err(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    error = compute_error_metric(score, len(y_true), z_score)
    return (error, error)


# %%
# Classifier
# ==========
# Now we use a Decision Tree to make predictions
#

classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X_train_processed, y_train_true)
y_test_pred = classifier.predict(X_test_processed)

# %%
# MetricFrame
# ===========
# Now we create a MetricFrame to generate the recall Wilson bounds and recall normal error
#

# Analyze metrics using MetricFrame
metrics_dict = {
    'Recall': recall_score,
    'Recall Bounds': recall_wilson,
    'Recall Error': recall_normal_err,
    'Accuracy': accuracy_score,
    'Accuracy Bounds': accuracy_wilson,
    'Accuracy Error': accuracy_normal_err,
}
metric_frame = MetricFrame(metrics_dict, y_test_true, y_test_pred, sensitive_features=test_set_sex)

# %%
# Plotting
# ========
# Plot metrics with (symmetric) error bars
# ----------------------------------------
plot_metric_frame(metric_frame, kind="scatter",
                  metrics=['Recall', 'Accuracy'],
                  error_bars=['Recall Error', 'Accuracy Error'])
plot_metric_frame(metric_frame, kind="bar", metrics='Recall', error_bars='Recall Error')

# %%
# Plot metrics with confidence intervals (possibly asymmetric)
# ------------------------------------------------------------
plot_metric_frame(metric_frame, kind="bar", metrics=[
                  'Recall', 'Accuracy'], conf_intervals=['Recall Bounds', 'Accuracy Bounds'])
plot_metric_frame(metric_frame, kind="scatter",
                  metrics='Recall', conf_intervals='Recall Bounds')

# %%
# Plot metrics without error bars
# -------------------------------
plot_metric_frame(metric_frame, kind="scatter", metrics=['Recall', 'Accuracy'])

# %%
# Plots all columns and treats them as metrics without error bars
# ---------------------------------------------------------------
plot_metric_frame(metric_frame, kind="bar")

# %%
# Plots no metrics (or should it return an error)
# -----------------------------------------------
# TODO: remove later, but something we should determine how to handle
plot_metric_frame(metric_frame, kind="bar", metrics=[])

# %%
# Custom Plot
# -----------
# Demonstrates how to customize the axes and then pass into `plot_metric_frame`

fig, axs = plt.subplots(*(1, 2), squeeze=False)
axs = axs.flatten()

axs[0].set_title("Custom Recall")
axs[1].set_title("Custom Accuracy")
axs[0].set_xlabel("Sensitive Feature")
axs[1].set_xlabel("Sensitive Feature")

# import matplotlib
# from matplotlib import cm
# sm = cm.ScalarMappable(norm=matplotlib.colors.Normalize(), cmap="Pastel1")
# cbar = plt.colorbar(sm)
# cbar.remove()
axs = plot_metric_frame(metric_frame, axs=axs, kind="scatter")


# %%
# Creating Custom Error Metrics
# =============================
# In this tutorial we used Normal intervals and Wilson bounds to demonstrate how to plot symmetric and asymmetric metrics, respectively.
#
# However, in order to support many different metrics, we reuse the MetricFrame functionality by writing a function
# that takes in y_true and y_pred, and returns the error metric:


def error_metric_function(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    # compute custom metric function here
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits_of_precision,
                    z_score)   # compute custom metric function here

    # returns the bounds in the format (lower_bound, upper_bound)
    return bounds
