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


from fairlearn.metrics import MetricFrame, ErrorPlotter, true_positive_rate, false_positive_rate, false_negative_rate, true_negative_rate, selection_rate, count
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

data = fetch_openml(data_id=1590, as_frame=True)
X = pd.get_dummies(data.data)
y_true = (data.target == '>50K') * 1
sex = data.data['sex']

# %%
# Error Metrics 
# -------------
# Now we define wilson bounds (asymmetric) and a symmetric error metric
#

z_score = 1.959964
digits = 4

def wilson(p, n, digits= digits, z = z_score):
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)))/np.sqrt(n)
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return (round(lower_bound, digits), round(upper_bound, digits))

def compute_error_metric(metric_value, sample_size, z_score):
    return z_score*np.sqrt(metric_value*(1.0-metric_value))/np.sqrt(sample_size)


def recall_wilson_lower_bound(y_t, y_p):
    assert len(y_t)==len(y_p)
    tn, fp, fn, tp  = confusion_matrix(y_t, y_p).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits, z_score)
    return bounds[0]

def recall_wilson_upper_bound(y_t, y_p):
    assert len(y_t)==len(y_p)
    tn, fp, fn, tp  = confusion_matrix(y_t, y_p).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits, z_score)
    return bounds[1]

def recall_wilson_lower_error(y_t, y_p):
    assert len(y_t)==len(y_p)
    tn, fp, fn, tp  = confusion_matrix(y_t, y_p).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits, z_score)
    return true_positive_rate(y_t, y_p) - bounds[0]

def recall_wilson_upper_error(y_t, y_p):
    assert len(y_t)==len(y_p)
    tn, fp, fn, tp  = confusion_matrix(y_t, y_p).ravel()
    bounds = wilson(tp/(tp+fn), tp + fn, digits, z_score)
    return bounds[1] - true_positive_rate(y_t, y_p)

def recall_err(y_t, y_p):
    assert len(y_t)==len(y_p)
    tn, fp, fn, tp  = confusion_matrix(y_t, y_p).ravel()
    error = compute_error_metric(tp/(tp+fn), tp + fn, z_score= z_score)
    return error

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
# ----------
# Now we create a metric frame
#

# Analyze metrics using MetricFrame
metrics = {
    # 'accuracy': accuracy_score,
    # 'precision': precision_score,
    'Recall': recall_score,
    'Recall lower bound': recall_wilson_lower_bound, 
    'Recall upper bound': recall_wilson_upper_bound, 
    'Recall lower error': recall_wilson_lower_error, 
    'Recall upper error': recall_wilson_upper_error, 
    'Recall Error': recall_err,
    # 'false positive rate': false_positive_rate,
    # 'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': count}
metric_frame = MetricFrame(metrics, y_true, y_pred, sensitive_features=sex)

# %%
# Error Plotting
# ----------
# Finally we plot the Recall metric with its error bounds
#
error_mapping = {
    "Recall": {
        "upper_bound": "Recall upper bound",
        "lower_bound": "Recall lower bound"
    }
}

ep = ErrorPlotter(metric_frame, error_mapping)
ep.plot_with_error("bar", metric="Recall", title="(Asymmetric) Absolute Bounds", capsize=10, colormap="Pastel1")
