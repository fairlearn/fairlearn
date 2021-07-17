# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from fairlearn.datasets import (
    fetch_adult,
    fetch_boston,
    fetch_bank_marketing,
    make_synthetic_dataset,
)

# =============================================


class TestFairlearnDataset:

    @pytest.mark.parametrize("as_frame", [True, False])
    @pytest.mark.parametrize("fetch_function", [fetch_adult, fetch_boston, fetch_bank_marketing])
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
    @pytest.mark.parametrize("fetch_function", [fetch_adult, fetch_boston, fetch_bank_marketing])
    def test_dataset_as_X_y(self, as_frame, fetch_function):
        X, y = fetch_function(as_frame=as_frame, return_X_y=True)
        assert X is not None
        assert isinstance(X, pd.DataFrame if as_frame else np.ndarray)
        assert y is not None
        assert isinstance(y, pd.Series if as_frame else np.ndarray)

    def test_synthetic_datasets(self):
        """Ensure that dataset creation is deterministic."""
        rng = np.random.RandomState(12345)
        X, y, gender = make_synthetic_dataset(random_state=rng)

        X_train, _, y_train, _, gender_train, _ = train_test_split(
            X, y, gender, test_size=0.3, random_state=rng
        )

        assert np.sum(X_train[0] < 0) == 8
        assert np.sum(y_train) == 699

        counts = {'Man': 0, 'Woman': 0, 'Other': 0, 'Unspecified': 0}
        for k in gender_train:
            counts[k] += 1
        expected_counts = {
            'Man': 340,
            'Woman': 353,
            'Other': 350,
            'Unspecified': 357
        }
        assert counts == expected_counts
