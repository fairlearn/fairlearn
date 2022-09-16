# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from fairlearn.metrics._annotated_metric_function import AnnotatedMetricFunction

from fairlearn.metrics._disaggregated_result import DisaggregatedResult
from fairlearn.metrics._bootstrap import (
    generate_single_bootstrap_sample,
    generate_bootstrap_samples,
    calculate_pandas_quantiles,
)

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
            random_state=185363,
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        assert da.overall.shape == target.overall.shape
        assert da.by_group.shape == target.by_group.shape
        # Hard to check values, but should make sure something changed
        assert any(da.overall != target.overall)
        assert any(da.by_group != target.by_group)

    def test_randomstate_stable(self):
        reference = generate_single_bootstrap_sample(
            random_state=124729,
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        comparison = generate_single_bootstrap_sample(
            random_state=124729,
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        # Should match exactly with same seed
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
            random_state=723548,
            data=basic_data,
            annotated_functions=metric2_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=["g_2"],
        )

        assert da.overall.shape == target.overall.shape
        assert da.by_group.shape == target.by_group.shape
        # Hard to check values, but should make sure something changed
        assert any(da.overall != target.overall)
        assert any(da.by_group != target.by_group)


class TestGenerateSamples:
    def test_smoke(self):
        da = DisaggregatedResult.create(
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        n_samples_wanted = 10
        target = generate_bootstrap_samples(
            random_state=142,
            n_samples=n_samples_wanted,
            data=basic_data,
            annotated_functions=metric_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=None,
        )

        assert isinstance(target, list)
        assert len(target) == n_samples_wanted

        for t in target:
            assert isinstance(t, DisaggregatedResult)
            assert da.overall.shape == t.overall.shape
            assert da.by_group.shape == t.by_group.shape
            # Make sure something changed
            assert any(da.overall != t.overall)
            assert any(da.by_group != t.by_group)

    def test_random_control(self):
        n_samples_wanted = 10
        seeds = [724435 for _ in range(n_samples_wanted)]

        target = generate_bootstrap_samples(
            random_state=seeds,
            n_samples=n_samples_wanted,
            data=basic_data,
            annotated_functions=metric2_dict,
            sensitive_feature_names=["g_1"],
            control_feature_names=["g_2"],
        )

        assert len(target) == n_samples_wanted
        for t in target:
            assert isinstance(t, DisaggregatedResult)
            # Make sure seed has been stable
            assert all(t.overall == target[0].overall)
            assert all(t.by_group == target[0].by_group)


class TestPandasQuantiles:
    def test_smoke_series(self):
        n_elements = 11
        name = 'My Series'
        index_val = 'My Value'

        # Create uniformly spaced data
        data = [pd.Series(data=x, name=name, index=[index_val]) for x in range(n_elements)]
        quantiles = [0.4, 0.5, 0.6]

        result = calculate_pandas_quantiles(quantiles=quantiles, bootstrap_samples=data)
        assert isinstance(result, pd.Series)
        assert result.name == name
        assert result.shape == (1,)
        assert np.array_equal(result[index_val], [4, 5, 6])

    def test_smoke_series2(self):
        n_elements = 11
        name = 'My Series'
        idx = ['a', 'b']

        # Create uniformly spaced data
        data = [pd.Series(data=[x, 2*x], name=name, index=idx) for x in range(n_elements)]
        quantiles = [0.4, 0.5, 0.6]

        result = calculate_pandas_quantiles(quantiles=quantiles, bootstrap_samples=data)
        assert isinstance(result, pd.Series)
        assert result.name == name
        assert result.shape == (2,)
        assert np.array_equal(result['a'], [4, 5, 6])
        assert np.array_equal(result['b'], [8, 10, 12])
