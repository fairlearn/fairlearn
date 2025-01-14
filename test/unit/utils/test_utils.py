# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

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
