# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics._annotated_metric_function import AnnotatedMetricFunction

from fairlearn.metrics._disaggregated_result import DisaggregatedResult
from fairlearn.metrics._bootstrap import generate_single_bootstrap_sample

from .data_for_test import g_1, g_2, y_p, y_t

basic_data = pd.DataFrame(data={"g_1": g_1, "g_2": g_2, "y_pred": y_p, "y_true": y_t})
metric_dict = {
    "recall": AnnotatedMetricFunction(
        func=skm.recall_score,
        name="recall_score",
        positional_argument_names=["y_true", "y_pred"],
    )
}
metric2_dict = {
    "recall": AnnotatedMetricFunction(
        func=skm.recall_score,
        name="recall_score",
        positional_argument_names=["y_true", "y_pred"],
    ),
    "accuracy": AnnotatedMetricFunction(
        func=skm.accuracy_score,
        name="accuracy_score",
        positional_argument_names=["y_true", "y_pred"],
    ),
}


class TestSingleSample:
    def test_smoke(self):
        da = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        target = generate_single_bootstrap_sample(
            random_state=128,
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        assert da.overall.shape == target.overall.shape
        assert da.by_group.shape == target.by_group.shape

    def test_randomstate_stable(self):
        reference = generate_single_bootstrap_sample(
            random_state=128,
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        comparison = generate_single_bootstrap_sample(
            random_state=128,
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        assert all(reference.overall == comparison.overall)
        assert all(reference.by_group == comparison.by_group)

    def test_multifunc_controlfeatures(self):
        da = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric2_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=["g_2"],
        )

        target = generate_single_bootstrap_sample(
            random_state=128,
            data=basic_data,
            annotated_functions=metric2_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=["g_2"],
        )

        assert da.overall.shape == target.overall.shape
        assert da.by_group.shape == target.by_group.shape
