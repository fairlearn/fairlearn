# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
"""Tests for the prejudice remover regularizer."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from fairlearn.linear_model import PrejudiceRemover
from fairlearn.metrics import demographic_parity_difference


def make_data(n_samples=400, seed=0, sensitive_effect=2.0):
    """Build data where the sensitive feature carries most of the signal."""
    rng = np.random.default_rng(seed)
    sensitive = rng.integers(0, 2, size=n_samples)
    X = rng.normal(size=(n_samples, 3))
    logits = 0.5 * X[:, 0] + sensitive_effect * sensitive
    y = (logits + rng.normal(scale=0.5, size=n_samples) > 1.0).astype(int)
    return X, y, sensitive


def test_eta_zero_matches_per_group_logistic_regression():
    X, y, sensitive = make_data()
    remover = PrejudiceRemover(eta=0.0, C=1.0).fit(X, y, sensitive_features=sensitive)

    assert remover.coef_.shape == (2, X.shape[1])
    assert remover.intercept_.shape == (2,)
    for group, value in enumerate(remover.sensitive_feature_values_):
        mask = sensitive == value
        reference = LogisticRegression(C=1.0, tol=1e-8, max_iter=1000).fit(X[mask], y[mask])
        np.testing.assert_allclose(remover.coef_[group], reference.coef_[0], rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            remover.intercept_[group], reference.intercept_[0], rtol=1e-4, atol=1e-4
        )


def test_prejudice_index_shrinks_as_eta_grows():
    X, y, sensitive = make_data()
    unpenalized = PrejudiceRemover(eta=0.0).fit(X, y, sensitive_features=sensitive)
    penalized = PrejudiceRemover(eta=10.0).fit(X, y, sensitive_features=sensitive)

    assert unpenalized.prejudice_index_ > 0
    assert penalized.prejudice_index_ < unpenalized.prejudice_index_


def test_larger_eta_reduces_selection_rate_gap():
    X, y, sensitive = make_data(sensitive_effect=3.0)
    unpenalized = PrejudiceRemover(eta=0.0).fit(X, y, sensitive_features=sensitive)
    penalized = PrejudiceRemover(eta=100.0).fit(X, y, sensitive_features=sensitive)

    def gap(estimator):
        predictions = estimator.predict(X, sensitive_features=sensitive)
        return demographic_parity_difference(y, predictions, sensitive_features=sensitive)

    assert gap(penalized) < gap(unpenalized)


def test_predict_proba_is_a_distribution_and_predict_is_its_argmax():
    X, y, sensitive = make_data()
    remover = PrejudiceRemover().fit(X, y, sensitive_features=sensitive)

    probabilities = remover.predict_proba(X, sensitive_features=sensitive)
    assert probabilities.shape == (X.shape[0], 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert remover.predict(X, sensitive_features=sensitive).tolist() == (
        remover.classes_[np.argmax(probabilities, axis=1)].tolist()
    )


def test_gradient_matches_finite_differences():
    X, y, sensitive = make_data(n_samples=60)
    remover = PrejudiceRemover(eta=0.7, C=2.0).fit(X, y, sensitive_features=sensitive)

    design = np.hstack([X, np.ones((X.shape[0], 1))])
    group_index = np.searchsorted(remover.sensitive_feature_values_, sensitive)
    weights = np.column_stack([remover.coef_, remover.intercept_]).ravel()

    _, analytic = remover._objective(weights, design, y.astype(float), group_index)

    step = 1e-6
    numeric = np.empty_like(weights)
    for index in range(weights.size):
        shifted = weights.copy()
        shifted[index] += step
        forward, _ = remover._objective(shifted, design, y.astype(float), group_index)
        shifted[index] -= 2 * step
        backward, _ = remover._objective(shifted, design, y.astype(float), group_index)
        numeric[index] = (forward - backward) / (2 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-6)


def test_string_labels_and_sensitive_values():
    X, y, sensitive = make_data()
    labels = np.where(y == 1, "yes", "no")
    groups = np.where(sensitive == 1, "group-b", "group-a")

    remover = PrejudiceRemover().fit(X, labels, sensitive_features=groups)
    predictions = remover.predict(X, sensitive_features=groups)

    assert remover.classes_.tolist() == ["no", "yes"]
    assert remover.sensitive_feature_values_.tolist() == ["group-a", "group-b"]
    assert set(np.unique(predictions)) <= {"no", "yes"}


def test_fit_without_intercept():
    X, y, sensitive = make_data()
    remover = PrejudiceRemover(eta=0.0, fit_intercept=False).fit(
        X, y, sensitive_features=sensitive
    )

    reference = LogisticRegression(C=1.0, fit_intercept=False, tol=1e-8, max_iter=1000).fit(
        X[sensitive == 0], y[sensitive == 0]
    )
    np.testing.assert_allclose(remover.coef_[0], reference.coef_[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(remover.intercept_, np.zeros(2))


def test_more_than_two_labels_raise():
    X, _, sensitive = make_data()
    y = np.arange(X.shape[0]) % 3

    with pytest.raises(ValueError, match="binary classification"):
        PrejudiceRemover().fit(X, y, sensitive_features=sensitive)


def test_sensitive_features_with_two_columns_raise():
    X, y, sensitive = make_data()
    two_columns = np.column_stack([sensitive, sensitive])

    with pytest.raises(ValueError, match="single feature"):
        PrejudiceRemover().fit(X, y, sensitive_features=two_columns)


def test_mismatched_lengths_raise():
    X, y, sensitive = make_data()

    with pytest.raises(ValueError, match="samples"):
        PrejudiceRemover().fit(X, y, sensitive_features=sensitive[:-1])


def test_missing_sensitive_values_raise():
    X, y, sensitive = make_data()
    groups = sensitive.astype(object)
    groups[0] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        PrejudiceRemover().fit(X, y, sensitive_features=groups)


def test_unseen_sensitive_value_at_predict_raises():
    X, y, sensitive = make_data()
    remover = PrejudiceRemover().fit(X, y, sensitive_features=sensitive)
    unseen = sensitive.copy()
    unseen[0] = 7

    with pytest.raises(ValueError, match="not seen during fit"):
        remover.predict_proba(X, sensitive_features=unseen)


def test_predict_before_fit_raises():
    X, _, sensitive = make_data()

    with pytest.raises(NotFittedError):
        PrejudiceRemover().predict(X, sensitive_features=sensitive)


@pytest.mark.parametrize("C", [0.0, -1.0])
def test_non_positive_C_raises(C):
    X, y, sensitive = make_data()

    with pytest.raises(ValueError, match="C must be strictly positive"):
        PrejudiceRemover(C=C).fit(X, y, sensitive_features=sensitive)


def test_negative_eta_raises():
    X, y, sensitive = make_data()

    with pytest.raises(ValueError, match="eta must be non-negative"):
        PrejudiceRemover(eta=-0.5).fit(X, y, sensitive_features=sensitive)
