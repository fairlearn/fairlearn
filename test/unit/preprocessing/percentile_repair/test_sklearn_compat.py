# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from __future__ import annotations

import narwhals.stable.v1 as nw
import numpy as np
import pytest

from fairlearn.preprocessing import PercentileRepairer
from test._sklearn_compat import parametrize_with_checks

pytestmark = pytest.mark.narwhals


EXPECTED_FAILED_CHECKS = {
    "PercentileRepairer": {
        "check_fit_idempotent": (
            "the check fits on part of the samples and transforms the rest, and with a "
            "continuous sensitive feature every held-out row is a group of its own, "
            "which PercentileRepairer has no distribution for"
        ),
    },
}


@parametrize_with_checks(
    [
        PercentileRepairer(sensitive_feature_ids=[0]),
        PercentileRepairer(sensitive_feature_ids=[0], repair_level=0.5),
    ],
    expected_failed_checks=lambda x: EXPECTED_FAILED_CHECKS.get(x.__class__.__name__, {}),
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_dataframe_and_numpy_agree(constructor) -> None:
    data = {"group": [0, 0, 0, 1, 1, 1], "score": [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]}
    X = constructor(data)

    X_tfm_df = PercentileRepairer(sensitive_feature_ids=["group"]).fit_transform(X)
    X_tfm_np = PercentileRepairer(sensitive_feature_ids=[0]).fit_transform(
        nw.from_native(X, eager_only=True).to_numpy()
    )

    np.testing.assert_array_equal(X_tfm_df, X_tfm_np)


def test_get_feature_names_out(constructor) -> None:
    data = {"group": [0.0, 0.0, 1.0, 1.0], "score": [1.0, 2.0, 3.0, 4.0]}
    X = constructor(data)
    repairer = PercentileRepairer(sensitive_feature_ids=["group"]).fit(X)

    names = repairer.get_feature_names_out()

    np.testing.assert_array_equal(names, ["group", "score"])
    assert len(names) == repairer.transform(X).shape[1]


def test_get_feature_names_out_for_arrays() -> None:
    X = np.arange(8, dtype=float).reshape(4, 2)
    repairer = PercentileRepairer(sensitive_feature_ids=[0]).fit(X)

    np.testing.assert_array_equal(repairer.get_feature_names_out(), ["x0", "x1"])
    np.testing.assert_array_equal(
        repairer.get_feature_names_out(["group", "score"]), ["group", "score"]
    )


def test_get_feature_names_out_validates_input_features() -> None:
    X = np.arange(8, dtype=float).reshape(4, 2)
    repairer = PercentileRepairer(sensitive_feature_ids=[0]).fit(X)

    with pytest.raises(ValueError, match="input_features should have length equal"):
        repairer.get_feature_names_out(["group"])


def test_get_feature_names_out_rejects_renamed_columns() -> None:
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"group": [0.0, 0.0, 1.0, 1.0], "score": [1.0, 2.0, 3.0, 4.0]})
    repairer = PercentileRepairer(sensitive_feature_ids=["group"]).fit(X)

    with pytest.raises(ValueError, match="not equal to the column names seen during fit"):
        repairer.get_feature_names_out(["group", "renamed"])


def test_transform_rejects_reordered_or_renamed_columns() -> None:
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"group": [0.0, 0.0, 1.0, 1.0], "score": [1.0, 2.0, 3.0, 4.0]})
    repairer = PercentileRepairer(sensitive_feature_ids=["group"]).fit(X)
    expected = repairer.transform(X)

    with pytest.raises(ValueError, match="columns and order seen during fit"):
        repairer.transform(X[["score", "group"]])

    np.testing.assert_array_equal(repairer.transform(X), expected)


def test_transform_rejects_a_different_number_of_features() -> None:
    X = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    repairer = PercentileRepairer(sensitive_feature_ids=[0]).fit(X)

    with pytest.raises(ValueError, match="X has 3 features, but PercentileRepairer"):
        repairer.transform(np.column_stack([X, X[:, 1]]))


def test_set_output_pandas() -> None:
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame({"group": [0.0, 0.0, 1.0, 1.0], "score": [1.0, 2.0, 3.0, 4.0]})
    repairer = PercentileRepairer(sensitive_feature_ids=["group"]).set_output(transform="pandas")

    X_tfm = repairer.fit_transform(X)

    assert isinstance(X_tfm, pd.DataFrame)
    assert list(X_tfm.columns) == ["group", "score"]
    np.testing.assert_array_equal(
        X_tfm.to_numpy(), PercentileRepairer(sensitive_feature_ids=["group"]).fit_transform(X)
    )
