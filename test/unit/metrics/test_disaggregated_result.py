# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics._annotated_metric_function import AnnotatedMetricFunction
from fairlearn.metrics._base_metrics import selection_rate
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
            str(e0.value) == "Invalid grouping function specified. Valid values are ['min', 'max']"
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
            str(e0.value) == "Invalid error value specified. Valid values are ['raise', 'coerce']"
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
            str(e0.value) == "Invalid error value specified. Valid values are ['raise', 'coerce']"
        )


@pytest.mark.parametrize(
    ["grouping_names", "expected"],
    [(None, pd.Series({"selection_rate": 0.5})), ([], pd.Series({"selection_rate": 0.5}))],
)
def test_apply_functions_with_no_grouping(grouping_names, expected):
    data = pd.DataFrame(
        {
            "y_pred": [1, 0, 1, 0, 0, 1],
            "y_true": [1, 1, 0, 1, 0, 0],
            "sensitive_feature": ["A", "A", "A", "B", "B", "B"],
        }
    )

    annotated_functions = {
        "selection_rate": AnnotatedMetricFunction(func=selection_rate, name="selection_rate")
    }

    result = DisaggregatedResult._apply_functions(
        data=data, annotated_functions=annotated_functions, grouping_names=grouping_names
    )

    pd.testing.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    ["grouping_names", "expected"],
    [
        (
            ["sensitive_feature"],
            pd.DataFrame(
                {"selection_rate": [2 / 3, 1 / 3]},
                index=pd.Index(["A", "B"], name="sensitive_feature"),
            ),
        ),
        (
            ["control_feature_1"],
            pd.DataFrame(
                {"selection_rate": [1 / 3, 2 / 3]},
                index=pd.Index(["X", "Y"], name="control_feature_1"),
            ),
        ),
        (
            ["control_feature_2", "sensitive_feature"],
            pd.DataFrame(
                {"selection_rate": [1.0, None, 0.5, 1 / 3]},
                index=pd.MultiIndex.from_product(
                    [("W", "Z"), ("A", "B")], names=["control_feature_2", "sensitive_feature"]
                ),
            ),
        ),
    ],
)
def test_apply_functions_with_grouping(grouping_names, expected):
    data = pd.DataFrame(
        {
            "y_pred": [1, 0, 1, 0, 0, 1],
            "y_true": [1, 1, 0, 1, 0, 0],
            "sensitive_feature": ["A", "A", "A", "B", "B", "B"],
            "control_feature_1": ["X", "X", "Y", "Y", "X", "Y"],
            "control_feature_2": ["Z", "Z", "W", "Z", "Z", "Z"],
        }
    )

    annotated_functions = {
        "selection_rate": AnnotatedMetricFunction(func=selection_rate, name="selection_rate")
    }

    result = DisaggregatedResult._apply_functions(
        data=data, annotated_functions=annotated_functions, grouping_names=grouping_names
    )

    pd.testing.assert_frame_equal(result, expected)
