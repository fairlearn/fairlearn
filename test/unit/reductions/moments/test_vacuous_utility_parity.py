"""Regression tests for vacuous utility-parity constraints."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from fairlearn.reductions import (
    ExponentiatedGradient,
    FalsePositiveRateParity,
    GridSearch,
    TruePositiveRateParity,
)


class ConstantClassifier(ClassifierMixin, BaseEstimator):
    """Clone-compatible one-class estimator accepting sample weights."""

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.constant_ = self.classes_[0]
        return self

    def predict(self, X):
        return np.full(len(X), self.constant_)


@pytest.mark.parametrize(
    ("constraint", "y"),
    [(TruePositiveRateParity(), np.zeros(4)), (FalsePositiveRateParity(), np.ones(4))],
)
def test_global_absent_event_has_vacuous_zero_constraints(constraint, y):
    """An absent conditioning event produces empty, safe constraint vectors."""
    X = np.arange(8).reshape(4, 2)
    sensitive = np.array([0, 0, 1, 1])
    constraint.load_data(X, y, sensitive_features=sensitive)

    assert constraint.index.empty
    assert constraint.gamma(lambda _: y).empty
    assert constraint.bound().empty
    assert constraint.project_lambda(pd.Series(dtype=float)).empty


@pytest.mark.parametrize("reduction", [ExponentiatedGradient, GridSearch])
@pytest.mark.parametrize(
    ("constraint", "y"),
    [(TruePositiveRateParity(), np.zeros(4)), (FalsePositiveRateParity(), np.ones(4))],
)
def test_vacuous_events_fit_reductions_end_to_end(reduction, constraint, y):
    """One-class absent events fit reductions and predict without constraints."""
    X = np.arange(8).reshape(4, 2)
    sensitive = np.array([0, 0, 1, 1])
    estimator = reduction(ConstantClassifier(), constraint)
    estimator.fit(X, y, sensitive_features=sensitive)
    assert estimator.predict(X).shape == y.shape
