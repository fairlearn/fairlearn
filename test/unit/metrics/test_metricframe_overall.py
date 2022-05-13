# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from fairlearn.metrics import MetricFrame

from .data_for_test import g_1, g_2, g_3, s_w, y_p, y_t

metric = [
    skm.recall_score,
    skm.precision_score,
    skm.accuracy_score,
    skm.balanced_accuracy_score,
]


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_0cf(metric_fn):
    target = MetricFrame(
        metrics=metric_fn, y_true=y_t, y_pred=y_p, sensitive_features=g_1
    )  # Sensitive features actually unused
    assert target._user_supplied_callable is True

    # Check user-visible
    expected = metric_fn(y_t, y_p)
    assert isinstance(target.overall, float)
    assert target.overall == expected


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_0cf_metric_dict(metric_fn):
    target = MetricFrame(
        metrics={metric_fn.__name__: metric_fn},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,
    )  # Sensitive features actually unusedle is False
    assert target._user_supplied_callable is False

    # Check user visible
    expected = metric_fn(y_t, y_p)
    assert isinstance(target.overall, pd.Series)
    assert len(target.overall) == 1
    assert target.overall[metric_fn.__name__] == expected


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_0cf_wgt(metric_fn):
    target = MetricFrame(
        metrics=metric_fn,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,  # Unused for overall
        sample_params={"sample_weight": s_w},
    )

    assert target._user_supplied_callable is True

    # Check user visible
    expected = metric_fn(y_t, y_p, sample_weight=s_w)
    assert isinstance(target.overall, float)
    assert target.overall == expected


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_0cf_wgt_metric_dict(metric_fn):
    target = MetricFrame(
        metrics={metric_fn.__name__: metric_fn},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_1,  # Unused for overall
        sample_params={metric_fn.__name__: {"sample_weight": s_w}},
    )
    assert target._user_supplied_callable is False

    # Check user visible
    expected = metric_fn(y_t, y_p, sample_weight=s_w)
    assert isinstance(target.overall, pd.Series)
    assert len(target.overall) == 1
    assert target.overall[metric_fn.__name__] == expected


def test_2m_0cf():
    funcs = {"recall": skm.recall_score, "prec": skm.precision_score}
    target = MetricFrame(
        metrics=funcs, y_true=y_t, y_pred=y_p, sensitive_features=g_1
    )  # Irrelevant for this test
    assert target._user_supplied_callable is False

    assert isinstance(target.overall, pd.Series)
    assert len(target.overall) == 2
    exp_recall = skm.recall_score(y_t, y_p)
    exp_prec = skm.precision_score(y_t, y_p)
    assert target.overall["recall"] == exp_recall
    assert target.overall["prec"] == exp_prec


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_1cf(metric_fn):
    target = MetricFrame(
        metrics=metric_fn,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_2,  # Unused here
        control_features=g_1,
    )
    assert target._user_supplied_callable is True

    mask_a = g_1 == "aa"
    mask_b = g_1 == "ba"
    exp_metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    exp_metric_b = metric_fn(y_t[mask_b], y_p[mask_b])

    # User visible features
    assert isinstance(target.overall, pd.Series)
    assert len(target.overall) == 2
    assert target.overall["aa"] == exp_metric_a
    assert target.overall["ba"] == exp_metric_b


@pytest.mark.parametrize("metric_fn", metric)
def test_1m_1cf_metric_dict(metric_fn):
    target = MetricFrame(
        metrics={metric_fn.__name__: metric_fn},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_2,
        control_features=g_1,
    )
    assert target._user_supplied_callable is False

    assert isinstance(target.overall, pd.DataFrame)
    assert target.overall.shape == (2, 1)
    assert np.array_equal(target.overall.index.names, ["control_feature_0"])
    mask_a = g_1 == "aa"
    mask_b = g_1 == "ba"
    exp_metric_a = metric_fn(y_t[mask_a], y_p[mask_a])
    exp_metric_b = metric_fn(y_t[mask_b], y_p[mask_b])
    assert target.overall[metric_fn.__name__]["aa"] == exp_metric_a
    assert target.overall[metric_fn.__name__]["ba"] == exp_metric_b


def test_1m_1cf_wgt():
    target = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_2,  # Unused
        control_features=g_1,
        sample_params={"sample_weight": s_w},
    )
    assert target._user_supplied_callable is True

    assert isinstance(target.overall, pd.Series)
    assert len(target.overall) == 2
    mask_a = g_1 == "aa"
    mask_b = g_1 == "ba"
    exp_recall_a = skm.recall_score(y_t[mask_a], y_p[mask_a], sample_weight=s_w[mask_a])
    exp_recall_b = skm.recall_score(y_t[mask_b], y_p[mask_b], sample_weight=s_w[mask_b])
    assert target.overall["aa"] == exp_recall_a
    assert target.overall["ba"] == exp_recall_b


def test_1m_1cf_wgt_metric_dict():
    target = MetricFrame(
        metrics={"recall": skm.recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_2,  # Unused
        control_features=g_1,
        sample_params={"recall": {"sample_weight": s_w}},
    )
    assert target._user_supplied_callable is False

    assert isinstance(target.overall, pd.DataFrame)
    assert target.overall.shape == (2, 1)
    assert np.array_equal(target.overall.index.names, ["control_feature_0"])
    mask_a = g_1 == "aa"
    mask_b = g_1 == "ba"
    exp_recall_a = skm.recall_score(y_t[mask_a], y_p[mask_a], sample_weight=s_w[mask_a])
    exp_recall_b = skm.recall_score(y_t[mask_b], y_p[mask_b], sample_weight=s_w[mask_b])
    assert target.overall["recall"]["aa"] == exp_recall_a
    assert target.overall["recall"]["ba"] == exp_recall_b


def test_1m_2cf():
    target = MetricFrame(
        metrics=skm.recall_score,
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_3,
        control_features=np.stack((g_1, g_2), axis=1),
    )
    assert target._user_supplied_callable is True

    assert isinstance(target.overall, pd.Series)
    assert len(target.overall) == 4
    assert np.array_equal(
        target.overall.index.names, ["control_feature_0", "control_feature_1"]
    )
    mask_a_f = np.logical_and((g_1 == "aa"), (g_2 == "f"))
    mask_a_g = np.logical_and((g_1 == "aa"), (g_2 == "g"))
    mask_b_f = np.logical_and((g_1 == "ba"), (g_2 == "f"))
    mask_b_g = np.logical_and((g_1 == "ba"), (g_2 == "g"))

    exp_a_f = skm.recall_score(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = skm.recall_score(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = skm.recall_score(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = skm.recall_score(y_t[mask_b_g], y_p[mask_b_g])

    assert target.overall[("aa", "f")] == exp_a_f
    assert target.overall[("aa", "g")] == exp_a_g
    assert target.overall[("ba", "f")] == exp_b_f
    assert target.overall[("ba", "g")] == exp_b_g


def test_1m_2cf_metric_dict():
    target = MetricFrame(
        metrics={"recall_score": skm.recall_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_3,
        control_features=np.stack((g_1, g_2), axis=1),
    )
    assert target._user_supplied_callable is False

    assert isinstance(target.overall, pd.DataFrame)
    assert target.overall.shape == (4, 1)
    assert np.array_equal(
        target.overall.index.names, ["control_feature_0", "control_feature_1"]
    )
    mask_a_f = np.logical_and((g_1 == "aa"), (g_2 == "f"))
    mask_a_g = np.logical_and((g_1 == "aa"), (g_2 == "g"))
    mask_b_f = np.logical_and((g_1 == "ba"), (g_2 == "f"))
    mask_b_g = np.logical_and((g_1 == "ba"), (g_2 == "g"))

    exp_a_f = skm.recall_score(y_t[mask_a_f], y_p[mask_a_f])
    exp_a_g = skm.recall_score(y_t[mask_a_g], y_p[mask_a_g])
    exp_b_f = skm.recall_score(y_t[mask_b_f], y_p[mask_b_f])
    exp_b_g = skm.recall_score(y_t[mask_b_g], y_p[mask_b_g])

    assert target.overall["recall_score"][("aa", "f")] == exp_a_f
    assert target.overall["recall_score"][("aa", "g")] == exp_a_g
    assert target.overall["recall_score"][("ba", "f")] == exp_b_f
    assert target.overall["recall_score"][("ba", "g")] == exp_b_g


def test_2m_2cf():
    two_group = pd.DataFrame(data=np.stack((g_1, g_2), axis=1), columns=["g_1", "g_2"])
    target = MetricFrame(
        metrics={"recall": skm.recall_score, "prec": skm.precision_score},
        y_true=y_t,
        y_pred=y_p,
        sensitive_features=g_3,  # Unused
        control_features=two_group,
    )

    assert isinstance(target.overall, pd.DataFrame)
    assert target.overall.shape == (4, 2)
    assert target.overall.index.names == ["g_1", "g_2"]
    mask_a_f = np.logical_and((g_1 == "aa"), (g_2 == "f"))
    mask_a_g = np.logical_and((g_1 == "aa"), (g_2 == "g"))
    mask_b_f = np.logical_and((g_1 == "ba"), (g_2 == "f"))
    mask_b_g = np.logical_and((g_1 == "ba"), (g_2 == "g"))

    recall_a_f = skm.recall_score(y_t[mask_a_f], y_p[mask_a_f])
    recall_a_g = skm.recall_score(y_t[mask_a_g], y_p[mask_a_g])
    recall_b_f = skm.recall_score(y_t[mask_b_f], y_p[mask_b_f])
    recall_b_g = skm.recall_score(y_t[mask_b_g], y_p[mask_b_g])
    assert target.overall["recall"][("aa", "f")] == recall_a_f
    assert target.overall["recall"][("aa", "g")] == recall_a_g
    assert target.overall["recall"][("ba", "f")] == recall_b_f
    assert target.overall["recall"][("ba", "g")] == recall_b_g

    prec_a_f = skm.precision_score(y_t[mask_a_f], y_p[mask_a_f])
    prec_a_g = skm.precision_score(y_t[mask_a_g], y_p[mask_a_g])
    prec_b_f = skm.precision_score(y_t[mask_b_f], y_p[mask_b_f])
    prec_b_g = skm.precision_score(y_t[mask_b_g], y_p[mask_b_g])
    assert target.overall["prec"][("aa", "f")] == prec_a_f
    assert target.overall["prec"][("aa", "g")] == prec_a_g
    assert target.overall["prec"][("ba", "f")] == prec_b_f
    assert target.overall["prec"][("ba", "g")] == prec_b_g
