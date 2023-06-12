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

QUANTILES = [0.05, 0.5, 0.95]


@pytest.fixture(scope="session")
def mf_1m_0cf():
    n_samples = 100
    target = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
    )
    target_boot = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_bootstrap_samples=n_samples,
        ci_quantiles=QUANTILES,
        bootstrap_random_state=13489623,
    )
    return target, target_boot


@pytest.fixture(scope="session")
def mf_1mdict_0cf():
    n_samples = 100
    target = MetricFrame(
        metrics={"recall": skm.recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
    )
    target_boot = MetricFrame(
        metrics={"recall": skm.recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_bootstrap_samples=n_samples,
        ci_quantiles=QUANTILES,
        bootstrap_random_state=13489623,
    )
    return target, target_boot


@pytest.fixture(scope="session")
def mf_1m_1cf():
    n_samples = 100
    target = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
    )
    target_boot = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_bootstrap_samples=n_samples,
        ci_quantiles=QUANTILES,
        bootstrap_random_state=13489623,
    )
    return target, target_boot


class TestOverallQuantiles:
    def test_1m_0cf(self, mf_1m_0cf):
        basic_mf = mf_1m_0cf[0]
        boot_mf = mf_1m_0cf[1]
        assert isinstance(boot_mf.overall, list)
        assert len(boot_mf.overall) == len(QUANTILES)
        # Overall value should be close to quantile 0.5
        assert boot_mf.overall[1] == pytest.approx(basic_mf.overall, abs=0.05)

    def test_1m_0cf_dict(self, mf_1mdict_0cf):
        basic_mf = mf_1mdict_0cf[0]
        boot_mf = mf_1mdict_0cf[1]
        assert isinstance(boot_mf.overall, list)
        assert len(boot_mf.overall) == len(QUANTILES)
        # Overall value should be close to quantile 0.5
        assert boot_mf.overall[1]["recall"] == pytest.approx(
            basic_mf.overall["recall"], abs=0.05
        )

    def test_1m_1_cf(self, mf_1m_1cf):
        basic_mf = mf_1m_1cf[0]
        boot_mf = mf_1m_1cf[1]
        
        assert isinstance(boot_mf.overall, list)
        assert len(boot_mf.overall) == len(QUANTILES)

        # Overall value should be close to quantile 0.5
        assert boot_mf.overall[1]["f"] == pytest.approx(
            basic_mf.overall["f"], abs=0.05
        )
        assert boot_mf.overall[1]["g"] == pytest.approx(
            basic_mf.overall["g"], abs=0.05
        )