# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
==========================================
RelaxedThresholdOptimizer with Census Data
==========================================
"""

# %%
# Load and preprocess the data set
# --------------------------------
# We download the data set using `fetch_adult` function in
# `fairlearn.datasets`. We start by importing the various modules we're going
# to use:
#

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.datasets import fetch_adult
from fairlearn.metrics import (
    MetricFrame,
    equalized_odds_difference,
    true_positive_rate,
    false_positive_rate,
    count,
    plot_model_comparison,
)

# %%
# We can now load and inspect the data by using the `fairlearn.datasets` module:

data = fetch_adult()
X_raw = data.data
Y = (data.target == ">50K") * 1
X_raw

# %%
# We are going to treat the sex of each individual as a sensitive feature
# (where 0 indicates female and 1 indicates male), and in this particular case
# we are going separate this feature out and drop it from the main data. We
# then perform some standard data preprocessing steps to convert the data into
# a format suitable for the ML algorithms

A = X_raw["sex"]
X = pd.get_dummies(X_raw)

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

le = LabelEncoder()
Y = le.fit_transform(Y)

# %%
# Finally, we split the data into training, validation, and test sets:
X_train, X_other, Y_train, Y_other, A_train, A_other = train_test_split(
    X_scaled, Y, A, test_size=0.4, random_state=0, stratify=Y,
)

# Split (X_other, Y_other, A_other) into validation and test
X_test, X_val, Y_test, Y_val, A_test, A_val = train_test_split(
    X_other, Y_other, A_other, test_size=0.5, random_state=0, stratify=Y_other,
)

# Work around indexing bug
X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
A_val = A_val.reset_index(drop=True)

# %%
# Training a fairness-unaware predictor
# -------------------------------------
# To show the effect of Fairlearn we will first train a standard ML predictor
# that does not incorporate fairness. For speed of demonstration, we use the
# simple :class:`sklearn.linear_model.LogisticRegression` class:

unmitigated_predictor = GradientBoostingClassifier(n_estimators=500)

# %%time
unmitigated_predictor.fit(X_train, Y_train)

# %%
# Compute predictions
y_test_pred_scores = unmitigated_predictor.predict_proba(X_test)[:, -1]
y_test_pred_binary = y_test_pred_scores >= 0.5      # threshold = 0.5

# %%
# We can start to assess the predictor's fairness using the `MetricFrame`:
metric_frame = MetricFrame(
    metrics={
        "accuracy": skm.accuracy_score,
        "true_positive_rate": true_positive_rate,
        "false_positive_rate": false_positive_rate,
        "count": count,
    },
    sensitive_features=A_test,
    y_true=Y_test,
    y_pred=y_test_pred_binary,
)
print(metric_frame.overall)
print(metric_frame.by_group)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[4, 1],
    legend=False,
    figsize=[12, 8],
    title="Accuracy and error-rates rate by group",
)


# %%
unmitigated_equalized_odds_diff = equalized_odds_difference(
    y_true=Y_test, y_pred=y_test_pred_binary, sensitive_features=A_test,
)

print(f"Equalized odds difference for unmitigated classifier: {unmitigated_equalized_odds_diff:.3}")

# %%
from fairlearn.postprocessing._cvxpy_threshold_optimizer import _RelaxedThresholdOptimizer

fair_clf = _RelaxedThresholdOptimizer(
    # predictor=unmitigated_predictor,    # TODO: use this when we no longer rely on callables
    # predict_method="predict_proba",
    predictor=lambda *args, **kwargs: unmitigated_predictor.predict(*args, **kwargs),
    predict_method="__call__",
    constraint="equalized_odds",
    tolerance=0,
)

# %%
fair_clf.fit(X_val, Y_val, sensitive_features=A_val)


# %%
y_test_pred_postprocessed = fair_clf.predict(X_test, sensitive_features=A_test)

# %%
postprocessed_equalized_odds_diff = equalized_odds_difference(
    y_true=Y_test, y_pred=y_test_pred_postprocessed, sensitive_features=A_test,
)

print(f"Equalized odds difference after postprocessing: {postprocessed_equalized_odds_diff:.3}")

# %%
# Add the unconstrained/unmitigated classifier predictions
all_model_predictions = {"unconstrained": y_test_pred_binary}


# Helper to get different thresholdings for different tolerance values
def compute_test_predictions_with_relaxed_constraints(tolerance: float) -> np.ndarray:
    # Instantiate
    clf = _RelaxedThresholdOptimizer(
        predictor=lambda *args, **kwargs: unmitigated_predictor.predict(*args, **kwargs),
        predict_method="__call__",
        constraint="equalized_odds",
        tolerance=tolerance,
        random_state=23,
    )

    # Fit
    clf.fit(X_train, Y_train, sensitive_features=A_train)

    return clf.predict(X_test, sensitive_features=A_test)


# Compute predictions at different levels of tolerance
all_model_predictions.update({
    f"train tolerance={tol:.1}": compute_test_predictions_with_relaxed_constraints(tol)
    for tol in np.arange(0, unmitigated_equalized_odds_diff, 1e-2)
})

# %%
# Plot all models in the fairness-accuracy landscape
plot_model_comparison(
    x_axis_metric=skm.accuracy_score,
    y_axis_metric=equalized_odds_difference,
    y_true=Y_test,
    y_preds=all_model_predictions,
    sensitive_features=A_test,
    point_labels=True,
    show_plot=True,
)
