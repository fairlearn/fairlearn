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
    target_boot = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=n_samples,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


@pytest.fixture(scope="session")
def mf_1mdict_0cf():
    n_samples = 100
    target_boot = MetricFrame(
        metrics={"recall": skm.recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=n_samples,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


@pytest.fixture(scope="session")
def mf_1m_1cf():
    n_samples = 100
    target_boot = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_boot=n_samples,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


@pytest.fixture(scope="session")
def mf_2m_1cf():
    n_samples = 100
    metric_dict = {"recall": skm.recall_score, "prec": skm.precision_score}
    target_boot = MetricFrame(
        metrics=metric_dict,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_boot=n_samples,
        ci_quantiles=QUANTILES,
        random_state=13489623,
    )
    return target_boot


class TestOverallQuantiles:
    def test_1m_0cf(self, mf_1m_0cf: MetricFrame):
        assert isinstance(mf_1m_0cf.overall_ci, list)
        assert len(mf_1m_0cf.overall_ci) == len(QUANTILES)
        assert mf_1m_0cf.ci_quantiles == QUANTILES
        # Overall value should be close to quantile 0.5
        assert mf_1m_0cf.overall_ci[1] == pytest.approx(mf_1m_0cf.overall, abs=0.05)

    def test_1m_0cf_dict(self, mf_1mdict_0cf):
        assert isinstance(mf_1mdict_0cf.overall_ci, list)
        assert len(mf_1mdict_0cf.overall_ci) == len(QUANTILES)
        assert mf_1mdict_0cf.ci_quantiles == QUANTILES
        # Overall value should be close to quantile 0.5
        assert mf_1mdict_0cf.overall_ci[1]["recall"] == pytest.approx(
            mf_1mdict_0cf.overall["recall"], abs=0.05
        )

    def test_1m_1_cf(self, mf_1m_1cf):
        assert isinstance(mf_1m_1cf.overall_ci, list)
        assert len(mf_1m_1cf.overall_ci) == len(QUANTILES)
        assert mf_1m_1cf.ci_quantiles == QUANTILES

        # Overall value should be close to quantile 0.5
        assert mf_1m_1cf.overall_ci[1]["f"] == pytest.approx(mf_1m_1cf.overall["f"], abs=0.05)
        assert mf_1m_1cf.overall_ci[1]["g"] == pytest.approx(mf_1m_1cf.overall["g"], abs=0.05)

    def test_2m_1_cf(self, mf_2m_1cf):
        assert isinstance(mf_2m_1cf.overall_ci, list)
        assert len(mf_2m_1cf.overall_ci) == len(QUANTILES)
        assert mf_2m_1cf.ci_quantiles == QUANTILES

        # Overall value should be close to quantile 0.5
        for m in ["recall", "prec"]:
            for cf in np.unique(g_2):
                assert mf_2m_1cf.overall_ci[1][m][cf] == pytest.approx(
                    mf_2m_1cf.overall[m][cf], abs=0.05
                )


class TestByGroupQuantiles:
    def test_1m_0cf(self, mf_1m_0cf: MetricFrame):
        assert isinstance(mf_1m_0cf.by_group_ci, list)
        assert len(mf_1m_0cf.by_group_ci) == len(QUANTILES)
        assert mf_1m_0cf.ci_quantiles == QUANTILES
        for g in np.unique(g_1):
            # Check median close to nominal
            assert mf_1m_0cf.by_group_ci[1][g] == pytest.approx(
                mf_1m_0cf.by_group[g], abs=0.05
            )

    def test_1m_0cf_dict(self, mf_1mdict_0cf: MetricFrame):
        assert isinstance(mf_1mdict_0cf.by_group_ci, list)
        assert len(mf_1mdict_0cf.by_group_ci) == len(QUANTILES)
        assert mf_1mdict_0cf.ci_quantiles == QUANTILES
        for g in np.unique(g_1):
            # Check median close to nominal
            assert mf_1mdict_0cf.by_group_ci[1]["recall"][g] == pytest.approx(
                mf_1mdict_0cf.by_group["recall"][g], abs=0.05
            )

    def test_1m_1_cf(self, mf_1m_1cf: MetricFrame):
        assert isinstance(mf_1m_1cf.by_group_ci, list)
        assert len(mf_1m_1cf.by_group_ci) == len(QUANTILES)
        assert mf_1m_1cf.ci_quantiles == QUANTILES
        for cf in np.unique(g_2):
            for g in np.unique(g_1):
                # Check median close to nominal
                assert mf_1m_1cf.by_group_ci[1][cf][g] == pytest.approx(
                    mf_1m_1cf.by_group[cf][g], abs=0.05
                )

    def test_2m_1_cf(self, mf_2m_1cf: MetricFrame):
        assert isinstance(mf_2m_1cf.by_group_ci, list)
        assert len(mf_2m_1cf.by_group_ci) == len(QUANTILES)
        assert mf_2m_1cf.ci_quantiles == QUANTILES

        for m in ["recall", "prec"]:
            for cf in np.unique(g_2):
                for g in np.unique(g_1):
                    # Check median close to nominal
                    assert mf_2m_1cf.by_group_ci[1][m][cf][g] == pytest.approx(
                        mf_2m_1cf.by_group[m][cf][g], abs=0.05
                    )
