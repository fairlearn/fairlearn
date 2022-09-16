# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics import MetricFrame

from .data_for_test import g_1, g_2, g_3, s_w, y_p, y_t


class TestOverallQuantiles:
    def test_1m_0cf(self):
        n_samples = 100
        quantiles = [0.2, 0.5, 0.9]
        target = MetricFrame(
            metrics=skm.recall_score,
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=g_1,
            n_bootstrap_samples=n_samples,
            bootstrap_random_state=13489623,
        )
        overall_quantiles = target.overall_quantiles(quantiles)
        assert isinstance(overall_quantiles, np.ndarray)
        assert overall_quantiles.shape==(len(quantiles),)
        # Overall value should be close to quantile 0.5
        assert target.overall == pytest.approx(overall_quantiles[1], abs=0.1)

    def test_1m_1_cf(self):
        n_samples = 100
        quantiles = [0.2, 0.5, 0.9]
        target = MetricFrame(
            metrics=skm.recall_score,
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=g_1,
            control_features=g_2,
            n_bootstrap_samples=n_samples,
            bootstrap_random_state=13489623,
        )
        overall_quantiles = target.overall_quantiles(quantiles)
        assert isinstance(overall_quantiles, pd.Series)
        assert overall_quantiles.name == 'recall_score'
        assert overall_quantiles.shape==(2,)
        # Overall value should be close to quantile 0.5
        assert target.overall['f'] == pytest.approx(overall_quantiles['f'][1], abs=0.1)
        assert target.overall['g'] == pytest.approx(overall_quantiles['g'][1], abs=0.1)
