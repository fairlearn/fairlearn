# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

import fairlearn.metrics.experimental as metrics
from .utils import _get_raw_GroupedMetric


class TestSingleFeature():
    def _get_raw_data(self):
        return ['a', 'a', 'b', 'c'], pd.Series(data=[0, 0, 1, 1])

    def _common_validations(self, result, expected_name):
        assert isinstance(result, list)
        assert len(result) == 1
        sf = result[0]
        assert isinstance(sf, metrics._group_feature.GroupFeature)
        assert sf.name == expected_name
        assert np.array_equal(sf.classes, ['a', 'b', 'c'])

    def test_single_list(self):
        raw_feature, y_true = self._get_raw_data()

        target = _get_raw_GroupedMetric()
        result = target._process_features("SF", raw_feature, y_true)
        self._common_validations(result, "SF 0")

    def test_single_series(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.Series(data=r_f, name="Some Series")

        target = _get_raw_GroupedMetric()
        result = target._process_features("Ignored", raw_feature, y_true)
        self._common_validations(result, "Some Series")

    def test_1d_array(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = np.asarray(r_f)

        target = _get_raw_GroupedMetric()
        result = target._process_features("CF", raw_feature, y_true)
        self._common_validations(result, "CF 0")

    def test_single_column_dataframe(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.DataFrame(data=r_f, columns=["My Feature"])

        target = _get_raw_GroupedMetric()
        result = target._process_features("Ignored", raw_feature, y_true)
        self._common_validations(result, "My Feature")

    def test_single_column_dataframe_unnamed(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.DataFrame(data=r_f)

        target = _get_raw_GroupedMetric()
        result = target._process_features("Unused", raw_feature, y_true)
        # If we don't specify names for the columns, then they are 'named' with integers
        self._common_validations(result, 0)


class TestTwoFeatures():
    def _get_raw_data(self):
        return ['a', 'a', 'b', 'c'], [5, 6, 6, 5], np.asarray([0, 1, 1, 0])

    def _common_validations(self, result, expected_names):
        assert isinstance(result, list)
        assert len(result) == 2
        for i in range(2):
            assert isinstance(result[i], metrics._group_feature.GroupFeature)
            assert result[i].name == expected_names[i]
        assert np.array_equal(result[0].classes, ['a', 'b', 'c'])
        assert np.array_equal(result[1].classes, [5, 6])

    def test_nested_list(self):
        a, b, y_true = self._get_raw_data()
        rf = [a, b]

        target = _get_raw_GroupedMetric()
        result = target._process_features('SF', rf, y_true)
        self._common_validations(result, ['SF 0', 'SF 1'])

    def test_2d_array(self):
        a, b, y_true = self._get_raw_data()
        # Specify dtype to avoid unwanted string conversion
        rf = np.asarray([a, b], dtype=np.object)

        target = _get_raw_GroupedMetric()
        result = target._process_features('CF', rf, y_true)
        self._common_validations(result, ['CF 0', 'CF 1'])

    def test_named_dataframe(self):
        cols = ["Col Alpha", "Col Num"]
        a, b, y_true = self._get_raw_data()

        rf = pd.DataFrame(data=zip(a, b), columns=cols)

        target = _get_raw_GroupedMetric()
        result = target._process_features('Ignored', rf, y_true)
        self._common_validations(result, cols)

    def test_unnamed_dataframe(self):
        a, b, y_true = self._get_raw_data()

        rf = pd.DataFrame(data=zip(a, b))

        target = _get_raw_GroupedMetric()
        result = target._process_features('Unused', rf, y_true)
        self._common_validations(result, [0, 1])

    def test_list_of_series(self):
        a, b, y_true = self._get_raw_data()

        rf = [pd.Series(data=a, name="Alpha"), pd.Series(data=b, name="Beta")]
        target = _get_raw_GroupedMetric()
        result = target._process_features('Unused', rf, y_true)
        self._common_validations(result, ['Alpha', 'Beta'])
