# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from collections import defaultdict
import pandas as pd
import numpy as np
import pytest

from fairlearn.datasets import (
    fetch_adult,
    fetch_boston,
    fetch_bank_marketing,
    SensitiveDatasetMaker,
    SensitiveFeature,
    SensitiveFeatureGroupConfig
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

    def test_simple_make_sensitive_classification(self):
        """Ensure that dataset creation is deterministic."""
        rng = np.random.RandomState(54321)
        gender_feature = SensitiveFeature(
            'gender', ['Man', 'Non-binary', 'Other', 'Unspecified', 'Woman'])
        dataset = SensitiveDatasetMaker(sensitive_features=[gender_feature],
                                        random_state=rng)
        X, y, features = dataset.make_sensitive_classification(n_samples=500)
        gender = features['gender']

        expected_counts = {
            'Man': 130,
            'Non-binary': 215,
            'Other': 164,
            'Unspecified': 272,
            'Woman': 237,
        }
        counts = defaultdict(int)
        for label, g in zip(y, gender):
            counts[g] += 1 if label else 0
        for k, v in expected_counts.items():
            assert v == counts[k]

        assert np.isclose(np.mean(X), -0.0047114)
        assert X.shape == (2500, 20)
        assert y.shape == (2500,)
        assert gender.shape == (2500,)

    def test_custom_make_sensitive_classification(self):
        """Ensure that custom dataset creation is deterministic."""
        rng = np.random.RandomState(54321)
        gender_feature = SensitiveFeature(
            'gender', ['Man', 'Non-binary', 'Other', 'Unspecified', 'Woman'])
        dataset = SensitiveDatasetMaker(sensitive_features=[gender_feature],
                                        random_state=rng)

        feature_config = {
            'Man': {'n_samples': 500, 'class_sep': 1},
            'Non-binary': {'n_samples': 200, 'class_sep': 3},
            'Other': {'n_samples': 900, 'class_sep': 0.5},
            'Unspecified': {'n_samples': 400, 'class_sep': 0.5},
            'Woman': {'n_samples': 300, 'class_sep': 2},
        }
        for key, config in feature_config.items():
            dataset.set_group_configs(config, gender=key)

        X, y, features = dataset.make_sensitive_classification(n_features=25)
        gender = features['gender']

        n_samples = np.sum([v['n_samples'] for v in feature_config.values()])
        assert set(pd.unique(gender)) == set(feature_config.keys())
        assert X.shape == (n_samples, 25)
        assert y.shape == (n_samples,)
        assert gender.shape == (n_samples,)
