# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import pytest

from fairlearn.datasets import (
    fetch_adult,
    fetch_bank_marketing,
    fetch_boston,
    fetch_credit_card,
    fetch_diabetes_hospital,
)

# =============================================


class TestFairlearnDataset:
    @pytest.mark.parametrize("as_frame", [True, False])
    @pytest.mark.parametrize(
        "fetch_function",
        [
            fetch_adult,
            fetch_bank_marketing,
            fetch_boston,
            fetch_credit_card,
            fetch_diabetes_hospital,
        ],
    )
    def test_dataset_as_bunch(self, as_frame, fetch_function):
        if as_frame is False and fetch_function is fetch_diabetes_hospital:
            msg = "fetch_diabetes_hospital will be skipped until dataset is fixed."
            pytest.skip(msg)
        dataset = fetch_function(as_frame=as_frame)
        assert dataset is not None
        assert dataset["data"].shape is not None
        assert isinstance(dataset["data"], pd.DataFrame if as_frame else np.ndarray)
        assert dataset["target"].shape is not None
        assert isinstance(dataset["target"], pd.Series if as_frame else np.ndarray)
        assert dataset["feature_names"] is not None
        assert isinstance(dataset["feature_names"], list)
        assert dataset["DESCR"] is not None
        assert isinstance(dataset["DESCR"], str)

    @pytest.mark.parametrize("as_frame", [True, False])
    @pytest.mark.parametrize(
        "fetch_function",
        [
            fetch_adult,
            fetch_bank_marketing,
            fetch_boston,
            fetch_credit_card,
            fetch_diabetes_hospital,
        ],
    )
    def test_dataset_as_X_y(self, as_frame, fetch_function):
        if as_frame is False and fetch_function is fetch_diabetes_hospital:
            msg = "fetch_diabetes_hospital will be skipped until dataset is fixed."
            pytest.skip(msg)
        X, y = fetch_function(as_frame=as_frame, return_X_y=True)
        assert X is not None
        assert isinstance(X, pd.DataFrame if as_frame else np.ndarray)
        assert y is not None
        assert isinstance(y, pd.Series if as_frame else np.ndarray)

    def test_fetch_diabetes_hospital_as_ndarray_raises_value_error(self):
        # Once this test no longer passes, the NOTE in the docstring of
        # fetch_diabetes_hospital regarding as_frame and the if clauses in
        # both dataset tests can be removed because the OpenML dataset has
        # been fixed.
        with pytest.raises(ValueError):
            fetch_diabetes_hospital(as_frame=False)
