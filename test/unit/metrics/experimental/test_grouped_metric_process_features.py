# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics


def _get_raw_GroupedMetric():
    # Gets an uninitialised GroupedMetric for testing purposes
    return metrics.GroupedMetric.__new__(metrics.GroupedMetric)


class TestSingleFeature():
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


class TestTwoFeatures():
    def _get_raw_data(self):
        return ['a', 'a', 'b', 'c'], [5, 6, 6, 5]

    def _common_validations(self, result, expected_names):
        assert isinstance(result, list)
        assert len(result) == 2
        for i in range(2):
            assert isinstance(result[i], metrics.SensitiveFeature)
            assert result[i].name == expected_names[i]
        assert np.array_equal(result[0].classes, ['a', 'b', 'c'])
        assert np.array_equal(result[1].classes, [5, 6])

    def test_nested_list(self):
        a, b = self._get_raw_data()
        rf = [a, b]

        target = _get_raw_GroupedMetric()
        result = target._process_features(rf, 4)
        self._common_validations(result, ['SF 0', 'SF 1'])

    def test_2d_array(self):
        a, b = self._get_raw_data()
        # Specify dtype to avoid unwanted string conversion
        rf = np.asarray([a, b], dtype=np.object)

        target = _get_raw_GroupedMetric()
        result = target._process_features(rf, 4)
        self._common_validations(result, ['SF 0', 'SF 1'])

    def test_named_dataframe(self):
        cols = ["Col Alpha", "Col Num"]
        a, b = self._get_raw_data()

        rf = pd.DataFrame(data=zip(a, b), columns=cols)

        target = _get_raw_GroupedMetric()
        result = target._process_features(rf, 4)
        self._common_validations(result, cols)

    def test_unnamed_dataframe(self):
        a, b = self._get_raw_data()

        rf = pd.DataFrame(data=zip(a, b))

        target = _get_raw_GroupedMetric()
        result = target._process_features(rf, 4)
        self._common_validations(result, [0, 1])
