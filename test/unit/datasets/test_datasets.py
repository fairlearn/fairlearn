# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

from fairlearn.datasets import fetch_adult, fetch_boston

# =============================================


class TestFairlearnDataset:
    def test_dataset_as_bunch_in_pandas_format(self):
        fetch_function_list = [fetch_adult, fetch_boston]
        for fetch_function in fetch_function_list:
            dataset = fetch_function(as_frame=True)
            assert dataset is not None
            assert dataset['data'].shape is not None
            assert isinstance(dataset['data'], pd.DataFrame)
            assert dataset['target'].shape is not None
            assert isinstance(dataset['target'], pd.Series)
            assert dataset['feature_names'] is not None
            assert isinstance(dataset['feature_names'], list)
            assert dataset['DESCR'] is not None
            assert isinstance(dataset['DESCR'], str)

    def test_dataset_as_bunch_in_numpy_format(self):
        fetch_function_list = [fetch_adult, fetch_boston]
        for fetch_function in fetch_function_list:
            dataset = fetch_function()
            assert dataset is not None
            assert dataset['data'].shape is not None
            assert isinstance(dataset['data'], np.ndarray)
            assert dataset['target'].shape is not None
            assert isinstance(dataset['target'], np.ndarray)
            assert dataset['feature_names'] is not None
            assert isinstance(dataset['feature_names'], list)
            assert dataset['DESCR'] is not None
            assert isinstance(dataset['DESCR'], str)

    def test_dataset_as_data_target_in_pandas_format(self):
        fetch_function_list = [fetch_adult, fetch_boston]
        for fetch_function in fetch_function_list:
            X, y = fetch_function(as_frame=True, return_X_y=True)
            assert X is not None
            assert isinstance(X, pd.DataFrame)
            assert y is not None
            assert isinstance(y, pd.Series)

    def test_dataset_as_data_target_in_numpy_format(self):
        fetch_function_list = [fetch_adult, fetch_boston]
        for fetch_function in fetch_function_list:
            X, y = fetch_function(as_frame=False, return_X_y=True)
            assert X is not None
            assert isinstance(X, np.ndarray)
            assert y is not None
            assert isinstance(y, np.ndarray)
