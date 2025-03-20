# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

import fairlearn.utils._input_validation as iv


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
