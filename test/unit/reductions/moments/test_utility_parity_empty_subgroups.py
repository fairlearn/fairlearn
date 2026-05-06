# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Regression tests for issue #543.

TruePositiveRateParity raises when a sensitive subgroup contains no
positive samples; FalsePositiveRateParity raises when a subgroup
contains no negative samples. Both should construct vacuous constraints
for the empty (group, event) cell instead of raising.

These tests are currently expected to fail on main; the fix replaces
``.size()`` with ``.count()`` in ``UtilityParity.load_data`` and emits
vacuous constraints for empty cells.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import (
    ExponentiatedGradient,
    FalsePositiveRateParity,
    TruePositiveRateParity,
)


def _data_with_empty_subgroup(empty_group_label):
    """Two sensitive groups; group ``a1`` is degenerate.

    Group ``a0``: 30 samples, balanced positives/negatives.
    Group ``a1``: 5 samples, all labelled ``empty_group_label``.

    Mirrors a realistic stratum in workforce-risk datasets where a
    small protected subgroup contains no instances of the target event.
    """
    rng = np.random.default_rng(0)

    n_a0 = 30
    a0_scores = rng.uniform(size=n_a0)
    a0_y = (a0_scores > 0.5).astype(int)

    n_a1 = 5
    a1_scores = rng.uniform(size=n_a1)
    a1_y = np.full(n_a1, empty_group_label, dtype=int)

    X = pd.DataFrame(
        {
            "score": np.concatenate([a0_scores, a1_scores]),
        }
    )
    y = np.concatenate([a0_y, a1_y])
    A = np.array(["a0"] * n_a0 + ["a1"] * n_a1)
    return X, y, A


def test_tprp_load_data_with_subgroup_of_all_negatives():
    """TPRP should not raise when a subgroup has no positives (#543)."""
    X, y, A = _data_with_empty_subgroup(empty_group_label=0)

    tprp = TruePositiveRateParity()
    tprp.load_data(X, y, sensitive_features=A)

    assert tprp.data_loaded
    assert tprp.total_samples == len(y)


def test_fprp_load_data_with_subgroup_of_all_positives():
    """FPRP should not raise when a subgroup has no negatives (#543)."""
    X, y, A = _data_with_empty_subgroup(empty_group_label=1)

    fprp = FalsePositiveRateParity()
    fprp.load_data(X, y, sensitive_features=A)

    assert fprp.data_loaded
    assert fprp.total_samples == len(y)


def test_expgrad_with_tprp_and_subgroup_of_all_negatives():
    """ExponentiatedGradient + TPRP end-to-end on degenerate stratum (#543)."""
    X, y, A = _data_with_empty_subgroup(empty_group_label=0)

    estimator = LogisticRegression(solver="liblinear")
    mitigator = ExponentiatedGradient(estimator, constraints=TruePositiveRateParity())
    mitigator.fit(X, y, sensitive_features=A)

    preds = mitigator.predict(X)
    assert preds.shape == (len(y),)


def test_expgrad_with_fprp_and_subgroup_of_all_positives():
    """ExponentiatedGradient + FPRP end-to-end on degenerate stratum (#543)."""
    X, y, A = _data_with_empty_subgroup(empty_group_label=1)

    estimator = LogisticRegression(solver="liblinear")
    mitigator = ExponentiatedGradient(estimator, constraints=FalsePositiveRateParity())
    mitigator.fit(X, y, sensitive_features=A)

    preds = mitigator.predict(X)
    assert preds.shape == (len(y),)
