# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

# These functions are borrowed from "test_metricframe_by_group.py"
# with additional checks for bootstrap related operations.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics import MetricFrame

from .data_for_test import y_t, y_p, s_w, g_1, g_2, g_3, g_4


metric = [
    skm.recall_score,
    #skm.precision_score,
    #skm.accuracy_score,
    #skm.balanced_accuracy_score,
]

ci_options = [0.95, 0.05, [0.2, 0.4, 0.6, 0.8]]
ci_methods = ["percentile", "bias-corrected"]

# Bootstrap test helper functions


def check_all_bootstrap_output_shapes(mf):
    # Check if all .*_ci outputs are same shape as regular outputs
    for ci_output in mf.overall_ci:
        assert mf.overall.shape == ci_output.shape
    for ci_output in mf.by_group_ci:
        assert mf.by_group.shape == ci_output.shape
    assert all(
        mf.group_min().shape == ci_output.shape for _, ci_output in mf.group_min_ci
    )
    assert all(
        mf.group_max().shape == ci_output.shape for _, ci_output in mf.group_max_ci
    )
    assert all(
        mf.difference(method="to_overall").shape == ci_output.shape
        for _, ci_output in mf.difference_overall_ci
    )
    assert all(
        mf.difference(method="between_groups").shape == ci_output.shape
        for _, ci_output in mf.difference_group_ci
    )
    assert all(
        mf.ratio(method="to_overall").shape == ci_output.shape
        for _, ci_output in mf.ratio_overall_ci
    )
    assert all(
        mf.ratio(method="between_groups").shape == ci_output.shape
        for _, ci_output in mf.ratio_group_ci
    )


def check_all_bootstrap_directions(mf):
    def check_direction(ci_outputs):
        # Iterate through all pairs of outputs and check that they are not decreasing
        return np.all([np.all(f <= s) for f, s in zip(ci_outputs, ci_outputs[1:])])

    assert check_direction(mf.overall_ci)
    assert check_direction(mf.by_group_ci)
    assert check_direction(mf.group_min_ci)
    assert check_direction(mf.group_max_ci)
    assert check_direction(mf.difference_overall_ci)
    assert check_direction(mf.difference_group_ci)
    assert check_direction(mf.ratio_overall_ci)
    assert check_direction(mf.ratio_group_ci)


@pytest.mark.parametrize("metric_fn", metric)
@pytest.mark.parametrize("ci_opt", ci_options)
@pytest.mark.parametrize("ci_method", ci_methods)
def test_1m_1sf_0cf(metric_fn, ci_opt, ci_method):
    target = MetricFrame(
        metrics=metric_fn,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci=ci_opt,
        ci_method=ci_method,
    )
    assert target._user_supplied_callable is True

    assert isinstance(target.by_group, pd.Series)
    assert len(target.by_group) == 2
    assert np.array_equal(target.by_group.index.names, ["sensitive_feature_0"])
    mask_a = g_1 == "aa"
    mask_b = g_1 == "ba"
    metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    metric_b = metric_fn(y_t[mask_b], y_p[mask_b])
    assert target.by_group["aa"] == metric_a
    assert target.by_group["ba"] == metric_b

    check_all_bootstrap_output_shapes(target)
    check_all_bootstrap_directions(target)


@pytest.mark.parametrize("metric_fn", metric)
@pytest.mark.parametrize("ci_opt", ci_options)
@pytest.mark.parametrize("ci_method", ci_methods)
def test_1m_1sf_0cf_metric_dict(metric_fn, ci_opt, ci_method):
    target = MetricFrame(
        metrics={metric_fn.__name__: metric_fn},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        n_boot=10,
        ci=ci_opt,
        ci_method=ci_method,
    )
    assert target._user_supplied_callable is False

    assert isinstance(target.by_group, pd.DataFrame)
    assert target.by_group.shape == (2, 1)
    assert np.array_equal(target.by_group.index.names, ["sensitive_feature_0"])
    mask_a = g_1 == "aa"
    mask_b = g_1 == "ba"
    metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    metric_b = metric_fn(y_t[mask_b], y_p[mask_b])
    assert target.by_group[metric_fn.__name__]["aa"] == metric_a
    assert target.by_group[metric_fn.__name__]["ba"] == metric_b

    check_all_bootstrap_output_shapes(target)
    check_all_bootstrap_directions(target)


@pytest.mark.parametrize("metric_fn", metric)
@pytest.mark.parametrize("ci_opt", ci_options)
@pytest.mark.parametrize("ci_method", ci_methods)
def test_1m_1sf_1cf(metric_fn, ci_opt, ci_method):
    target = MetricFrame(
        metrics=metric_fn,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_boot=10,
        ci=ci_opt,
        ci_method=ci_method,
    )
    assert target._user_supplied_callable is True

    assert isinstance(target.by_group, pd.Series)
    assert len(target.by_group) == 4
    assert np.array_equal(
        target.by_group.index.names, ["control_feature_0", "sensitive_feature_0"]
    )

    mask_a_f = np.logical_and((g_1 == "aa"), (g_2 == "f"))
    mask_a_g = np.logical_and((g_1 == "aa"), (g_2 == "g"))
    mask_b_f = np.logical_and((g_1 == "ba"), (g_2 == "f"))
    mask_b_g = np.logical_and((g_1 == "ba"), (g_2 == "g"))

    exp_a_f = metric_fn(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = metric_fn(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = metric_fn(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = metric_fn(y_t[mask_b_g], y_p[mask_b_g])
    assert target.by_group[("f", "aa")] == exp_a_f
    assert target.by_group[("f", "ba")] == exp_b_f
    assert target.by_group[("g", "aa")] == exp_a_g
    assert target.by_group[("g", "ba")] == exp_b_g

    check_all_bootstrap_output_shapes(target)
    check_all_bootstrap_directions(target)


@pytest.mark.parametrize("metric_fn", metric)
@pytest.mark.parametrize("ci_opt", ci_options)
@pytest.mark.parametrize("ci_method", ci_methods)
def test_1m_1sf_1cf_metric_dict(metric_fn, ci_opt, ci_method):
    target = MetricFrame(
        metrics={metric_fn.__name__: metric_fn},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
        control_features=g_2,
        n_boot=10,
        ci=ci_opt,
        ci_method=ci_method,
    )
    assert target._user_supplied_callable is False

    assert isinstance(target.by_group, pd.DataFrame)
    assert target.by_group.shape == (4, 1)
    assert np.array_equal(
        target.by_group.index.names, ["control_feature_0", "sensitive_feature_0"]
    )

    mask_a_f = np.logical_and((g_1 == "aa"), (g_2 == "f"))
    mask_a_g = np.logical_and((g_1 == "aa"), (g_2 == "g"))
    mask_b_f = np.logical_and((g_1 == "ba"), (g_2 == "f"))
    mask_b_g = np.logical_and((g_1 == "ba"), (g_2 == "g"))

    exp_a_f = metric_fn(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = metric_fn(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = metric_fn(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = metric_fn(y_t[mask_b_g], y_p[mask_b_g])
    assert target.by_group[metric_fn.__name__][("f", "aa")] == exp_a_f
    assert target.by_group[metric_fn.__name__][("f", "ba")] == exp_b_f
    assert target.by_group[metric_fn.__name__][("g", "aa")] == exp_a_g
    assert target.by_group[metric_fn.__name__][("g", "ba")] == exp_b_g

    check_all_bootstrap_output_shapes(target)
    check_all_bootstrap_directions(target)


@pytest.mark.parametrize("ci_opt", ci_options)
@pytest.mark.parametrize("ci_method", ci_methods)
def test_2m_2sf_2cf(ci_opt, ci_method):
    funcs = {"recall": skm.recall_score, "prec": skm.precision_score}
    s_p = {"recall": {"sample_weight": s_w}}
    sf = np.stack([g_1, g_3], axis=1)
    cf = {"cf1": g_2, "cf2": g_4}

    target = MetricFrame(
        metrics=funcs,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=sf,
        control_features=cf,
        sample_params=s_p,
        n_boot=10,
        ci=ci_opt,
        ci_method=ci_method,
    )
    assert target._user_supplied_callable is False

    assert isinstance(target.by_group, pd.DataFrame)
    assert target.by_group.shape == (16, 2)
    assert np.array_equal(
        target.by_group.index.names,
        ["cf1", "cf2", "sensitive_feature_0", "sensitive_feature_1"],
    )

    # Only check some isolated results, rather than all 32
    mask_a_f = np.logical_and((g_1 == "aa"), (g_2 == "f"))
    mask_b_g = np.logical_and((g_1 == "ba"), (g_2 == "g"))
    mask_k_q = np.logical_and((g_3 == "kk"), (g_4 == "q"))

    mask_f_q_a_k = np.logical_and(mask_a_f, mask_k_q)
    recall_f_q_a_k = skm.recall_score(
        y_t[mask_f_q_a_k], y_p[mask_f_q_a_k], sample_weight=s_w[mask_f_q_a_k]
    )
    prec_f_q_a_k = skm.precision_score(y_t[mask_f_q_a_k], y_p[mask_f_q_a_k])
    assert target.by_group["recall"][("f", "q", "aa", "kk")] == recall_f_q_a_k
    assert target.by_group["prec"][("f", "q", "aa", "kk")] == prec_f_q_a_k

    mask_g_q_b_k = np.logical_and(mask_b_g, mask_k_q)
    recall_g_q_b_k = skm.recall_score(
        y_t[mask_g_q_b_k], y_p[mask_g_q_b_k], sample_weight=s_w[mask_g_q_b_k]
    )
    prec_g_q_b_k = skm.precision_score(y_t[mask_g_q_b_k], y_p[mask_g_q_b_k])
    assert target.by_group["recall"][("g", "q", "ba", "kk")] == recall_g_q_b_k
    assert target.by_group["prec"][("g", "q", "ba", "kk")] == prec_g_q_b_k

    check_all_bootstrap_output_shapes(target)
    check_all_bootstrap_directions(target)
