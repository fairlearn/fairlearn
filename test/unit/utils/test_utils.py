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
        assert isinstance(X_update, pd.DataFrame)
        assert isinstance(y_update, pd.Series)
        assert isinstance(y_update, pd.Series)
        assert cf_update is None

        assert np.array_equal(X, X_update)
        assert np.array_equal(y, y_update)
        assert np.array_equal(sf, sf_update)
