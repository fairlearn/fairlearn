# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.metrics import group_accuracy_score, group_roc_auc_score
from fairlearn.metrics._group_metric_set import _process_feature_to_integers
from fairlearn.metrics._group_metric_set import _process_sensitive_features
from fairlearn.metrics._group_metric_set import create_group_metric_set

from test.unit.input_convertors import conversions_for_1d


class TestProcessFeatureToInteger:
    @pytest.mark.parametrize("transform_feature", conversions_for_1d)
    def test_smoke(self, transform_feature):
        f = transform_feature([1, 2, 1, 2])

        names, f_integer = _process_feature_to_integers(f)
        assert isinstance(names, list)
        assert isinstance(f_integer, list)
        assert np.array_equal(names, ["1", "2"])
        assert np.array_equal(f_integer, [0, 1, 0, 1])

    @pytest.mark.parametrize("transform_feature", conversions_for_1d)
    def test_strings(self, transform_feature):
        f = transform_feature(['b', 'a', 'a', 'b'])

        names, f_integer = _process_feature_to_integers(f)
        assert isinstance(names, list)
        assert isinstance(f_integer, list)
        assert np.array_equal(names, ["a", "b"])
        assert np.array_equal(f_integer, [1, 0, 0, 1])


class TestProcessSensitiveFeatures:
    @pytest.mark.parametrize("transform_feature", conversions_for_1d)
    def test_smoke(self, transform_feature):
        sf_name = "My SF"
        sf_vals = transform_feature([1, 3, 3, 1])

        sf = {sf_name: sf_vals}
        result = _process_sensitive_features(sf)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]['featureBinName'] == sf_name
        assert result[0]['binVector'] == [0, 1, 1, 0]
        assert result[0]['binLabels'] == ["1", "3"]

    def test_result_is_sorted(self):
        sf_vals = [1, 2, 3, 1]

        sf = {"b": sf_vals, "a": sf_vals, "c": sf_vals}
        result = _process_sensitive_features(sf)
        assert isinstance(result, list)
        assert len(result) == 3
        for r in result:
            assert r['binVector'] == [0, 1, 2, 0]
            assert r['binLabels'] == ['1', '2', '3']
        result_names = [r['featureBinName'] for r in result]
        assert result_names == ["a", "b", "c"]
