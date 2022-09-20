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

    def test_1m_1_cf(self, mf_1m_1cf):
        quantiles = [0.2, 0.5, 0.9]
        overall_quantiles = mf_1m_1cf.overall_quantiles(quantiles)
        assert isinstance(overall_quantiles, pd.Series)
        assert overall_quantiles.name == "recall_score"
        assert overall_quantiles.shape == (2,)
        # Overall value should be close to quantile 0.5
        assert mf_1m_1cf.overall["f"] == pytest.approx(
            overall_quantiles["f"][1], abs=0.05
        )
        assert mf_1m_1cf.overall["g"] == pytest.approx(
            overall_quantiles["g"][1], abs=0.05
        )


class TestByGroupQuantiles:
    def test_1m_0cf(self, mf_1m_0cf):
        quantiles = [0.2, 0.5, 0.9]
        by_group_quantiles = mf_1m_0cf.by_group_quantiles(quantiles)
        assert isinstance(by_group_quantiles, pd.Series)
        for g in np.unique(g_1):
            curr = by_group_quantiles[g]
            assert isinstance(curr, np.ndarray)
            assert curr.shape == (3,)
            # Check median close to nominal
            assert curr[1] == pytest.approx(mf_1m_0cf.by_group[g], abs=0.05)

    def test_1m_1cf(self, mf_1m_1cf):
        quantiles = [0.2, 0.5, 0.9]
        by_group_quantiles = mf_1m_1cf.by_group_quantiles(quantiles)
        assert isinstance(by_group_quantiles, pd.Series)
        for cf in np.unique(g_2):
            for g in np.unique(g_1):
                curr = by_group_quantiles[cf][g]
                assert isinstance(curr, np.ndarray)
                assert curr.shape == (3,)
                # Check median close to nominal
                assert curr[1] == pytest.approx(mf_1m_1cf.by_group[cf][g], abs=0.05)



class TestGroupMax:
    def test_1m_0cf(self, mf_1m_0cf):
        quantiles = [0.2, 0.5, 0.9]

        baseline = mf_1m_0cf.group_max()
        ci_vals = mf_1m_0cf.group_max(quantiles=quantiles)
        assert isinstance(ci_vals, np.ndarray)
        assert ci_vals.shape == (len(quantiles),)
        # Median should be close to baseline
        assert baseline == pytest.approx(ci_vals[1], abs=0.05)
        assert ci_vals[0] < ci_vals[1]
        assert ci_vals[1] < ci_vals[2]

    def test_1m_1cf(self, mf_1m_1cf):
        quantiles = [0.2, 0.5, 0.9]

        baseline = mf_1m_1cf.group_max()
        ci_vals = mf_1m_1cf.group_max(quantiles=quantiles)
        assert isinstance(ci_vals, pd.Series)
        assert ci_vals.shape == (2,)
        # Median should be close to baseline in each case
        for g in np.unique(g_2):
            assert baseline[g] == pytest.approx(ci_vals[g][1], abs=0.1)
