# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics import MetricFrame

from .data_for_test import g_1, g_2, y_p, y_t

# Have fixtures so that tests can be specific without
# constantly recomputing the same bootstrapping


@pytest.fixture(scope="session")
def mf_1m_0cf():
    n_samples = 100
    target = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_bootstrap_samples=n_samples,
        ci_quantiles=[0.05, 0.5,0.95],
        bootstrap_random_state=13489623,
    )
    return target


@pytest.fixture(scope="session")
def mf_1mdict_0cf():
    n_samples = 100
    target = MetricFrame(
        metrics={"recall": skm.recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_bootstrap_samples=n_samples,
        ci_quantiles=[0.05, 0.5,0.95],
        bootstrap_random_state=13489623,
    )
    return target


@pytest.fixture(scope="session")
def mf_1m_1cf():
    n_samples = 100
    target = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_bootstrap_samples=n_samples,
        ci_quantiles=[0.05, 0.5,0.95],
        bootstrap_random_state=13489623,
    )
    return target


class TestOverallQuantiles:
    def test_1m_0cf(self, mf_1m_0cf):
        quantiles = [0.2, 0.5, 0.9]
        overall_quantiles = mf_1m_0cf.overall_quantiles(quantiles)
        assert isinstance(overall_quantiles, np.ndarray)
        assert overall_quantiles.shape == (len(quantiles),)
        # Overall value should be close to quantile 0.5
        assert mf_1m_0cf.overall == pytest.approx(overall_quantiles[1], abs=0.05)
