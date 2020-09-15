# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics


def _get_raw_GroupedMetric():
    # Gets an uninitialised GroupedMetric for testing purposes
    return metrics.GroupedMetric.__new__(metrics.GroupedMetric)


class TestSingleFeatures():
    def _get_raw_data(self):
        return ['a', 'a', 'b', 'c']

    def _common_validations(self, result, expected_name):
        assert isinstance(result, list)
        assert len(result) == 1
        sf = result[0]
        assert isinstance(sf, metrics.SensitiveFeature)
        assert sf.name == expected_name
        assert np.array_equal(sf.classes, ['a', 'b', 'c'])

    def test_single_list(self):
        raw_feature = self._get_raw_data()

        target = _get_raw_GroupedMetric()
        result = target._process_features(raw_feature, len(raw_feature))
        self._common_validations(result, "SF 0")

    def test_single_series(self):
        raw_feature = pd.Series(data=self._get_raw_data(), name="Some Series")

        target = _get_raw_GroupedMetric()
        result = target._process_features(raw_feature, len(raw_feature))
        self._common_validations(result, "Some Series")

    def test_1d_array(self):
        raw_feature = np.asarray(self._get_raw_data())

        target = _get_raw_GroupedMetric()
        result = target._process_features(raw_feature, len(self._get_raw_data()))
        self._common_validations(result, "SF 0")

    def test_single_column_dataframe(self):
        raw_feature = pd.DataFrame(data=self._get_raw_data(), columns=["My Feature"])

        target = _get_raw_GroupedMetric()
        result = target._process_features(raw_feature, len(self._get_raw_data()))
        self._common_validations(result, "My Feature")

    def test_single_column_dataframe_unnamed(self):
        raw_feature = pd.DataFrame(data=self._get_raw_data())

        target = _get_raw_GroupedMetric()
        result = target._process_features(raw_feature, len(self._get_raw_data()))
        # If we don't specify names for the columns, then they are 'named' with integers
        self._common_validations(result, 0)
