# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import pytest

from fairlearn.datasets import (
    fetch_adult,
    fetch_boston,
    fetch_bank_marketing,
    fetch_credit_card,
)

# =============================================


class TestFairlearnDataset:

    @pytest.mark.parametrize("as_frame", [True, False])
    @pytest.mark.parametrize("fetch_function", [fetch_adult, fetch_boston, fetch_bank_marketing, fetch_credit_card])
    def test_dataset_as_bunch(self, as_frame, fetch_function):
        dataset = fetch_function(as_frame=as_frame)
        assert dataset is not None
        assert dataset['data'].shape is not None
        assert isinstance(dataset['data'], pd.DataFrame if as_frame else np.ndarray)
        assert dataset['target'].shape is not None
        assert isinstance(dataset['target'], pd.Series if as_frame else np.ndarray)
        assert dataset['feature_names'] is not None
        assert isinstance(dataset['feature_names'], list)
        assert dataset['DESCR'] is not None
        assert isinstance(dataset['DESCR'], str)

    @pytest.mark.parametrize("as_frame", [True, False])
    @pytest.mark.parametrize("fetch_function", [fetch_adult, fetch_boston, fetch_bank_marketing, fetch_credit_card])
    def test_dataset_as_X_y(self, as_frame, fetch_function):
        X, y = fetch_function(as_frame=as_frame, return_X_y=True)
        assert X is not None
        assert isinstance(X, pd.DataFrame if as_frame else np.ndarray)
        assert y is not None
        assert isinstance(y, pd.Series if as_frame else np.ndarray)
