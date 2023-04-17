# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

import fairlearn.metrics as metrics

from .utils import _get_raw_MetricFrame


class TestSingleFeature:
    def _get_raw_data(self):
        return ["a", "a", "b", "c"], pd.Series(data=[0, 0, 1, 1])

    def _common_validations(self, result, expected_name):
        assert isinstance(result, list)
        assert len(result) == 1
        sf = result[0]
        assert isinstance(sf, metrics._group_feature.GroupFeature)
        assert sf.name_ == expected_name
        assert np.array_equal(sf.classes_, ["a", "b", "c"])

    def test_single_list(self):
        raw_feature, y_true = self._get_raw_data()

        target = _get_raw_MetricFrame()
        result = target._process_features("SF", raw_feature, y_true)
        self._common_validations(result, "SF0")

    def test_single_series(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.Series(data=r_f, name="Some Series")

        target = _get_raw_MetricFrame()
        result = target._process_features("Ignored", raw_feature, y_true)
        self._common_validations(result, "Some Series")

    def test_single_series_integer_name(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.Series(data=r_f, name=0)

        target = _get_raw_MetricFrame()
        msg = "Series name must be a string. Value '0' was of type <class 'int'>"
        with pytest.raises(ValueError) as execInfo:
            _ = target._process_features("Ignored", raw_feature, y_true)
        assert execInfo.value.args[0] == msg

    def test_1d_array(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = np.asarray(r_f)

        target = _get_raw_MetricFrame()
        result = target._process_features("CF", raw_feature, y_true)
        self._common_validations(result, "CF0")

    def test_single_column_dataframe(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.DataFrame(data=r_f, columns=["My Feature"])

        target = _get_raw_MetricFrame()
        result = target._process_features("Ignored", raw_feature, y_true)
        self._common_validations(result, "My Feature")

    def test_single_column_dataframe_unnamed(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.DataFrame(data=r_f)

        target = _get_raw_MetricFrame()
        msg = (
            "DataFrame column names must be strings. Name '0' is of type <class 'int'>"
        )
        with pytest.raises(ValueError) as execInfo:
            _ = target._process_features("Unused", raw_feature, y_true)
        assert execInfo.value.args[0] == msg

    def test_single_dict(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = {"Mine!": r_f}

        target = _get_raw_MetricFrame()
        result = target._process_features("Ignored", raw_feature, y_true)
        self._common_validations(result, "Mine!")

    def test_single_column_dataframe_integer_name(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = pd.DataFrame(data=r_f, columns=[0])

        target = _get_raw_MetricFrame()
        msg = (
            "DataFrame column names must be strings. Name '0' is of type <class"
            " 'numpy.int64'>"
        )
        with pytest.raises(ValueError) as execInfo:
            _ = target._process_features("Unused", raw_feature, y_true)
        assert execInfo.value.args[0] == msg

    def test_from_dict_failure(self):
        r_f, y_true = self._get_raw_data()
        raw_feature = {"Mine!": np.asarray(r_f).reshape(-1, 1)}
        target = _get_raw_MetricFrame()
        msg = (
            "DataFrame.from_dict() failed on sensitive features. "
            "Please ensure each array is strictly 1-D. "
            "The __cause__ field of this exception may contain further information."
        )
        with pytest.raises(ValueError) as ve:
            _ = target._process_features("Unused", raw_feature, y_true)
        assert msg == ve.value.args[0]
        assert ve.value.__cause__ is not None


class TestTwoFeatures:
    def _get_raw_data(self):
        return ["a", "a", "b", "c"], [5, 6, 6, 5], np.asarray([0, 1, 1, 0])

    def _common_validations(self, result, expected_names):
        assert isinstance(result, list)
        assert len(result) == 2
        for i in range(2):
            assert isinstance(result[i], metrics._group_feature.GroupFeature)
            assert result[i].name_ == expected_names[i]
        assert np.array_equal(result[0].classes_, ["a", "b", "c"])
        assert np.array_equal(result[1].classes_, [5, 6])

    def test_nested_list(self):
        a, b, y_true = self._get_raw_data()
        rf = [a, b]

        target = _get_raw_MetricFrame()
        msg = "Feature lists must be of scalar types"
        with pytest.raises(ValueError) as execInfo:
            _ = target._process_features("SF", rf, y_true)
        assert execInfo.value.args[0] == msg

    def test_2d_array(self):
        a, b, y_true = self._get_raw_data()
        # Specify dtype to avoid unwanted string conversion
        rf = np.asarray([a, b], dtype=object).transpose()

        target = _get_raw_MetricFrame()
        result = target._process_features("CF", rf, y_true)
        self._common_validations(result, ["CF0", "CF1"])

    def test_named_dataframe(self):
        cols = ["Col Alpha", "Col Num"]
        a, b, y_true = self._get_raw_data()

        rf = pd.DataFrame(data=zip(a, b), columns=cols)

        target = _get_raw_MetricFrame()
        result = target._process_features("Ignored", rf, y_true)
        self._common_validations(result, cols)

    def test_unnamed_dataframe(self):
        a, b, y_true = self._get_raw_data()

        rf = pd.DataFrame(data=zip(a, b))

        target = _get_raw_MetricFrame()
        msg = (
            "DataFrame column names must be strings. Name '0' is of type <class 'int'>"
        )
        with pytest.raises(ValueError) as execInfo:
            _ = target._process_features("Unused", rf, y_true)
        assert execInfo.value.args[0] == msg

    def test_list_of_series(self):
        a, b, y_true = self._get_raw_data()

        rf = [pd.Series(data=a, name="Alpha"), pd.Series(data=b, name="Beta")]
        target = _get_raw_MetricFrame()
        msg = "Feature lists must be of scalar types"
        with pytest.raises(ValueError) as execInfo:
            _ = target._process_features("Unused", rf, y_true)
        assert execInfo.value.args[0] == msg

    def test_dictionary(self):
        a, b, y_true = self._get_raw_data()

        rf = {"Alpha": a, "Beta": b}
        target = _get_raw_MetricFrame()
        result = target._process_features("Unused", rf, y_true)
        self._common_validations(result, ["Alpha", "Beta"])
