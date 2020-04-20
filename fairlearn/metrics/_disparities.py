# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Metrics for measuring disparity."""

from ._metrics_engine import (
    selection_rate_difference,
    selection_rate_ratio,
    true_positive_rate_difference,
    true_positive_rate_ratio,
    false_positive_rate_difference,
    false_positive_rate_ratio)


def demographic_parity_difference(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity difference."""
    return selection_rate_difference(
        y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight)


def demographic_parity_ratio(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity ratio."""
    return selection_rate_ratio(
        y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight)


def equalized_odds_difference(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity difference."""
    return max(
        true_positive_rate_difference(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight),
        false_positive_rate_difference(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight))    


def equalized_odds_ratio(y_true, y_pred, *, sensitive_features, sample_weight=None):
    r"""Calculate the demographic parity ratio."""
    return min(
        true_positive_rate_ratio(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight),
        false_positive_rate_ratio(
            y_true, y_pred, sensitive_features=sensitive_features, sample_weight=sample_weight))    
