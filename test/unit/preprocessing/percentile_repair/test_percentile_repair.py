# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from __future__ import annotations

import numpy as np
import pytest

from fairlearn.preprocessing import PercentileRepairer


def test_repairs_every_group_to_the_median_distribution() -> None:
    # Two groups with the same size but disjoint ranges. The repair reads each
    # value's percentile rank off the median of the two quantile functions, so
    # both groups end up with the same three values.
    X = np.array(
        [
            [0, 1000],
            [0, 1100],
            [0, 1200],
            [1, 1300],
            [1, 1400],
            [1, 1500],
        ]
    )

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    np.testing.assert_array_equal(X_tfm[:, 0], [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(X_tfm[:, 1], [1150, 1250, 1350, 1150, 1250, 1350])


def test_repair_keeps_the_row_order() -> None:
    X = np.array([[1, 5], [0, 1], [1, 6], [0, 2]])

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    np.testing.assert_array_equal(X_tfm[:, 0], [1, 0, 1, 0])


def test_repair_is_monotone_inside_each_group() -> None:
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 2, 200)
    values = np.where(groups == 0, rng.normal(10, 3, 200), rng.normal(20, 1, 200))
    X = np.column_stack([groups, values])

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    for group in (0, 1):
        rows = X[:, 0] == group
        order = np.argsort(X[rows, 1], kind="stable")
        repaired = X_tfm[rows, 1][order]
        assert np.all(np.diff(repaired) >= 0)


def test_unequal_group_sizes_are_mapped_through_the_median_quantiles() -> None:
    # Group 0 holds 1..4, group 1 holds 10 and 20. The median quantile function
    # takes the rank of each value within its own group and evaluates both
    # groups' quantiles there.
    X = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 10], [1, 20]])

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    np.testing.assert_array_equal(X_tfm[:, 1], [5.5, 6, 11.5, 12, 5.5, 11.5])


def test_ties_stay_tied() -> None:
    X = np.array([[0, 1], [0, 1], [0, 3], [1, 10], [1, 10], [1, 30]])

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    assert X_tfm[0, 1] == X_tfm[1, 1]
    assert X_tfm[3, 1] == X_tfm[4, 1]
    assert X_tfm[0, 1] < X_tfm[2, 1]


@pytest.mark.parametrize("repair_level", [0, 0.0])
def test_repair_level_zero_leaves_the_data_unchanged(repair_level) -> None:
    X = np.array([[0, 1], [0, 2], [1, 10], [1, 20]])

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0], repair_level=repair_level).fit_transform(
        X
    )

    np.testing.assert_array_equal(X_tfm, X.astype(float))


def test_partial_repair_interpolates_between_the_original_and_the_repair() -> None:
    X = np.array([[0, 1000], [0, 1100], [0, 1200], [1, 1300], [1, 1400], [1, 1500]])
    full = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    half = PercentileRepairer(sensitive_feature_ids=[0], repair_level=0.5).fit_transform(X)

    np.testing.assert_array_equal(half[:, 1], (X[:, 1] + full[:, 1]) / 2)


def test_a_single_group_is_left_unchanged() -> None:
    # With one group the median distribution is the group's own distribution, so
    # every value maps back to itself.
    X = np.array([[1, 5], [1, 7], [1, 9]])

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    np.testing.assert_array_equal(X_tfm[:, 1], [5, 7, 9])


def test_sensitive_column_and_other_columns_are_both_handled() -> None:
    X = np.array([[0, 1, 100], [0, 2, 200], [1, 10, 100], [1, 20, 200]])

    X_tfm = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(X)

    # The sensitive column is passed through, and every other column is repaired
    # on its own: column 1 differs between the groups and is mapped onto the
    # median of the two distributions, while column 2 already agrees between the
    # groups and comes back unchanged.
    np.testing.assert_array_equal(X_tfm[:, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(X_tfm[:, 1], [5.5, 11, 5.5, 11])
    np.testing.assert_array_equal(X_tfm[:, 2], [100, 200, 100, 200])


def test_values_outside_the_fitted_range_are_mapped_to_the_ends() -> None:
    repairer = PercentileRepairer(sensitive_feature_ids=[0]).fit(
        np.array([[0, 1], [0, 2], [0, 3], [1, 10], [1, 20], [1, 30]])
    )
    below = repairer.transform(np.array([[0, -100]]))
    above = repairer.transform(np.array([[1, 100]]))

    assert below[0, 1] == repairer.transform(np.array([[0, 1]]))[0, 1]
    assert above[0, 1] == repairer.transform(np.array([[1, 30]]))[0, 1]


def test_transform_needs_a_fitted_estimator() -> None:
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        PercentileRepairer(sensitive_feature_ids=[0]).transform(np.array([[0, 1]]))


def test_a_group_not_seen_during_fit_is_rejected() -> None:
    repairer = PercentileRepairer(sensitive_feature_ids=[0]).fit(
        np.array([[0, 1], [0, 2], [1, 10], [1, 20]])
    )

    with pytest.raises(ValueError, match=r"group\(s\) \[2\.0\] that were not seen during fit"):
        repairer.transform(np.array([[2, 5]]))


def test_exactly_one_sensitive_feature_is_required() -> None:
    X = np.array([[0, 1, 2], [0, 3, 4]])

    with pytest.raises(ValueError, match="must name exactly one sensitive feature, got 0"):
        PercentileRepairer().fit(X)
    with pytest.raises(ValueError, match="must name exactly one sensitive feature, got 2"):
        PercentileRepairer(sensitive_feature_ids=[0, 1]).fit(X)


def test_repair_level_is_validated() -> None:
    X = np.array([[0, 1], [0, 2], [1, 3], [1, 4]])

    with pytest.raises(ValueError, match="repair_level must be between 0 and 1"):
        PercentileRepairer(sensitive_feature_ids=[0], repair_level=1.5).fit(X)
    with pytest.raises(ValueError, match="repair_level must be between 0 and 1"):
        PercentileRepairer(sensitive_feature_ids=[0], repair_level=-0.1).fit(X)


@pytest.mark.parametrize("column", [0, 1])
def test_missing_values_are_rejected(column) -> None:
    X = np.array([[0, 1], [0, 2], [1, 3], [1, 4]], dtype=float)
    X[2, column] = np.nan

    with pytest.raises(ValueError, match="contains NaN"):
        PercentileRepairer(sensitive_feature_ids=[0]).fit(X)


def test_non_numeric_values_are_rejected() -> None:
    pd = pytest.importorskip("pandas")
    data = pd.DataFrame({"group": [0, 0, 1, 1], "score": ["a", "b", "c", "d"]})

    with pytest.raises(ValueError, match="can only repair numeric columns"):
        PercentileRepairer(sensitive_feature_ids=["group"]).fit(data)


def test_an_unknown_column_is_rejected() -> None:
    X = np.array([[0, 1], [0, 2]])

    with pytest.raises(ValueError, match=r"Columns \[2\] not found in the input data"):
        PercentileRepairer(sensitive_feature_ids=[2]).fit(X)


def test_a_1d_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array instead"):
        PercentileRepairer(sensitive_feature_ids=[0]).fit(np.array([0.0, 1.0, 2.0]))
