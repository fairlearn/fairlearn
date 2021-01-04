# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.metrics import plot_disparities_in_performance, \
    plot_disparities_in_metric
from fairlearn.metrics._group_metric_set import BINARY_CLASSIFICATION_METRICS, \
    REGRESSION_METRICS
import pandas as pd
import pytest
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier


# copied verbatim from quickstart
def quickstart_setup():
    data = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(data.data)
    y_true = (data.target == '>50K') * 1
    sex = data.data['sex']

    classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    classifier.fit(X, y_true)
    y_pred = classifier.predict(X)
    return (y_true, y_pred, sex)


# These are just smoke tests.  For ways to improve testing,
# see https://github.com/fairlearn/fairlearn/pull/289#issuecomment-587170978
def test_plot_disparities_in_performance():
    (y_true, y_pred, sensitive_features) = quickstart_setup()
    plot_disparities_in_performance(y_true, y_pred, sensitive_features, show_plot=False)


@pytest.mark.parametrize(
    "metric", list(BINARY_CLASSIFICATION_METRICS.values()) +
    list(REGRESSION_METRICS.values()))
def test_plot_disparities_in_metric(metric):
    (y_true, y_pred, sensitive_features) = quickstart_setup()
    plot_disparities_in_metric(metric, y_true, y_pred, sensitive_features, show_plot=False)
