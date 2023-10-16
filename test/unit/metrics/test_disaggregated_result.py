# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics._annotated_metric_function import AnnotatedMetricFunction

from fairlearn.metrics._disaggregated_result import DisaggregatedResult

from .data_for_test import g_1, y_p, y_t

basic_data = pd.DataFrame(data={"g_1": g_1, "y_pred": y_p, "y_true": y_t})
metric_dict = {
    "recall": AnnotatedMetricFunction(
        func=skm.recall_score,
        name="recall_score",
        positional_argument_names=["y_true", "y_pred"],
    )
}

# This is not a comprehensive set of tests for DisaggregatedResult; those are mainly
# covered through the MetricFrame tests (DisaggregatedResult having been extracted
# from MetricFrame). For now, these tests fill in some holes in existing testing


class TestErrorMessages:
    def test_bad_grouping(self):
        target = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )
        with pytest.raises(ValueError) as e0:
            _ = target.apply_grouping("bad_func")
        assert (
            str(e0.value)
            == "Invalid grouping function specified. Valid values are ['min', 'max']"
        )

    def test_bad_difference_method(self):
        target = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )
        with pytest.raises(ValueError) as e0:
            _ = target.difference(control_feature_names=None, method="bad_func")
        assert str(e0.value) == "Unrecognised method 'bad_func' in difference() call"

    def test_bad_difference_errors(self):
        target = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )
        with pytest.raises(ValueError) as e0:
            _ = target.difference(
                control_feature_names=None, method="between_groups", errors="bad_option"
            )
        assert (
            str(e0.value)
            == "Invalid error value specified. Valid values are ['raise', 'coerce']"
        )

    def test_bad_ratio_method(self):
        target = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )
        with pytest.raises(ValueError) as e0:
            _ = target.ratio(control_feature_names=None, method="bad_func")
        assert str(e0.value) == "Unrecognised method 'bad_func' in ratio() call"

    def test_bad_ratio_errors(self):
        target = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )
        with pytest.raises(ValueError) as e0:
            _ = target.ratio(
                control_feature_names=None, method="between_groups", errors="bad_option"
            )
        assert (
            str(e0.value)
            == "Invalid error value specified. Valid values are ['raise', 'coerce']"
        )
