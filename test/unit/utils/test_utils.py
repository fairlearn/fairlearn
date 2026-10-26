# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import fairlearn.utils._input_validation as iv
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import BoundedGroupLoss, DemographicParity, EqualizedOdds, ZeroOneLoss


class TestValidateAndReformatInput:
    def test_smoke(self):
        # Regression test for Issue #898
        X = pd.DataFrame.from_dict({"alpha": ["a", "a", "b"], "beta": [1, 2, 1]})
        sf = np.asarray(["C", "D", "C"])
        y = np.asarray([0, 0, 1])

        X_update, y_update, sf_update, cf_update = iv._validate_and_reformat_input(
            X=X, y=y, sensitive_features=sf
        )
        assert isinstance(y_update, pd.Series)
        assert isinstance(sf_update, pd.Series)
        assert cf_update is None

        assert isinstance(X_update, pd.DataFrame)
        assert np.array_equal(X, X_update)
        assert np.array_equal(y, y_update)
        assert np.array_equal(sf, sf_update)

    def test_only_X(self):
        # Regression test for Issue #898
        X = pd.DataFrame.from_dict({"alpha": ["a", "a", "b"], "beta": [1, 2, 1]})

        X_update, _, _, _ = iv._validate_and_reformat_input(
            X=X, expect_y=False, expect_sensitive_features=False
        )
        assert isinstance(X_update, pd.DataFrame)

    def test_cf(self):
        # Regression test for Issue #898
        X = pd.DataFrame.from_dict({"alpha": ["a", "a", "b"], "beta": [1, 2, 1]})
        sf = np.asarray(["C", "D", "C"])
        y = np.asarray([0, 0, 1])
        cf = pd.DataFrame.from_dict({"gamma": ["c", "d", "d"], "delta": [0, 0, 0]})

        X_update, y_update, sf_update, cf_update = iv._validate_and_reformat_input(
            X=X, y=y, sensitive_features=sf, control_features=cf
        )
        assert isinstance(y_update, pd.Series)
        assert isinstance(cf_update, pd.Series)

        assert isinstance(X_update, pd.DataFrame)
        assert np.array_equal(X, X_update)
        assert np.array_equal(y, y_update)
        assert np.array_equal(sf, sf_update)


def test_validate_and_reformat_input_allow_ndims_greater_than_2() -> None:
    X = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    sf = np.array([1, 2])

    X_update, _, _, _ = iv._validate_and_reformat_input(X=X, sensitive_features=sf, expect_y=False)

    np.testing.assert_array_equal(X, X_update)


@pytest.mark.parametrize(
    ("input_data", "expected"),
    [
        (np.array([["A", "B"], ["C", "D"]]), np.array(["A,B", "C,D"])),
        (
            np.array(
                [
                    ["test\\with\\backslash", "normal"],
                    ["normal", "test,with,,separator"],
                    ["both\\types,test", "value"],
                ]
            ),
            np.array(
                [
                    "test\\\\with\\\\backslash,normal",
                    "normal,test\\,with\\,\\,separator",
                    "both\\\\types\\,test,value",
                ]
            ),
        ),
    ],
)
def test_merge_columns(input_data, expected):
    result = iv._merge_columns(input_data)
    np.testing.assert_array_equal(result, expected)


def test_merge_columns_value_error():
    with pytest.raises(
        ValueError, match=r"Received argument of type list instead of expected numpy\.ndarray"
    ):
        iv._merge_columns([["A", "1"], ["B", "2"]])


@pytest.mark.parametrize(
    "y", [[], np.asarray([]), pd.Series(dtype="float64"), pd.DataFrame(), None]
)
def test_validate_and_reformat_input_empty_y(y):
    """Test that _validate_and_reformat_input raises as expected when y is expected, but
    passed as an empty list, nd.array, series or dataframe or None."""
    X = pd.DataFrame.from_dict({"alpha": ["a", "a", "b"], "beta": [1, 2, 1]})

    with pytest.raises(ValueError, match=iv._MESSAGE_Y_NONE):
        X, y, _, _ = iv._validate_and_reformat_input(
            X=X, y=y, expect_y=True, expect_sensitive_features=False
        )


def _feature_with_missing_value(kind, missing):
    values = ["a", "b", missing, "a"]
    if kind == "list":
        return values
    if kind == "ndarray":
        return np.asarray(values, dtype=object)
    if kind == "series":
        return pd.Series(values)
    # two columns, which would otherwise be merged into a single string column
    return pd.DataFrame({"f0": values, "f1": ["c", "d", "c", "d"]})


@pytest.mark.parametrize("kind", ["list", "ndarray", "series", "dataframe"])
@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
@pytest.mark.parametrize("feature_name", ["sensitive_features", "control_features"])
def test_validate_and_reformat_input_missing_feature_values(kind, missing, feature_name):
    X = pd.DataFrame({"alpha": [1, 2, 3, 4]})
    y = np.asarray([0, 1, 0, 1])
    features = {"sensitive_features": ["a", "b", "a", "b"]}
    features[feature_name] = _feature_with_missing_value(kind, missing)

    msg = iv._FEATURE_HAS_MISSING_VALUES_ERROR_MESSAGE.format(feature_name)
    with pytest.raises(ValueError, match=re.escape(msg)):
        iv._validate_and_reformat_input(X=X, y=y, **features)


@pytest.mark.parametrize(
    "fit",
    [
        lambda X, y, sf: DemographicParity().load_data(X, y, sensitive_features=sf),
        lambda X, y, sf: EqualizedOdds().load_data(X, y, sensitive_features=sf),
        lambda X, y, sf: BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1).load_data(
            X, y, sensitive_features=sf
        ),
        lambda X, y, sf: ThresholdOptimizer(
            estimator=LogisticRegression(), predict_method="predict_proba"
        ).fit(X, y, sensitive_features=sf),
    ],
)
def test_mitigation_rejects_missing_sensitive_feature_values(fit):
    # None used to be dropped silently from the groups in the reductions moments
    X = pd.DataFrame({"alpha": np.arange(8)})
    y = np.asarray([0, 1] * 4)
    sf = np.asarray(["a", "b"] * 4, dtype=object)
    sf[0] = None

    with pytest.raises(ValueError, match="sensitive_features contains missing values"):
        fit(X, y, sf)
