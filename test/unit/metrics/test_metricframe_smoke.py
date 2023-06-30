# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from test.unit.input_convertors import conversions_for_1d

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics

# Bring in some pre-prepared input arrays
from .data_for_test import g_1, g_2, g_3, g_4, s_w, y_p, y_t


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_basic(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    g_f = pd.DataFrame(data=g_4, columns=["My feature"])
    target = metrics.MetricFrame(
        metrics=skm.recall_score,
        y_true=transform_y_t(y_t),
        y_pred=transform_y_p(y_p),
        sensitive_features=g_f,
    )
    # Check on ci_quantiles
    assert target.ci_quantiles is None

    # Check on the indices properties
    assert target.control_levels is None
    assert isinstance(target.sensitive_levels, list)
    assert target.sensitive_levels == ["My feature"]

    # Check we have correct return types
    assert isinstance(target.overall, float)
    assert isinstance(target.by_group, pd.Series)

    # Check we have expected number of elements
    assert len(target.by_group) == 2
    assert np.array_equal(target.by_group.index.names, ["My feature"])

    recall_overall = skm.recall_score(y_t, y_p)
    assert target.overall == recall_overall

    mask_p = g_4 == "pp"
    mask_q = g_4 == "q"
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    assert target.by_group["pp"] == recall_p
    assert target.by_group["q"] == recall_q

    target_mins = target.group_min()
    assert isinstance(target_mins, float)
    assert target_mins == min(recall_p, recall_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, float)
    assert target_maxes == max(recall_p, recall_q)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_basic_metric_dict(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    g_f = pd.DataFrame(data=g_4, columns=["My feature"])
    target = metrics.MetricFrame(
        metrics={"recall": skm.recall_score},
        y_true=transform_y_t(y_t),
        y_pred=transform_y_p(y_p),
        sensitive_features=g_f,
    )
    # Check on ci_quantiles
    assert target.ci_quantiles is None

    # Check on the indices properties
    assert target.control_levels is None
    assert isinstance(target.sensitive_levels, list)
    assert target.sensitive_levels == ["My feature"]

    # Check we have correct return types
    assert isinstance(target.overall, pd.Series)
    assert isinstance(target.by_group, pd.DataFrame)

    # Check we have expected number of elements
    assert len(target.overall) == 1
    assert target.by_group.shape == (2, 1)
    assert np.array_equal(target.by_group.index.names, ["My feature"])

    recall_overall = skm.recall_score(y_t, y_p)
    assert target.overall["recall"] == recall_overall

    mask_p = g_4 == "pp"
    mask_q = g_4 == "q"
    recall_p = skm.recall_score(y_t[mask_p], y_p[mask_p])
    recall_q = skm.recall_score(y_t[mask_q], y_p[mask_q])
    assert target.by_group["recall"]["pp"] == recall_p
    assert target.by_group["recall"]["q"] == recall_q

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.Series)
    assert len(target_mins) == 1
    assert target_mins["recall"] == min(recall_p, recall_q)

    target_maxes = target.group_max()
    assert isinstance(target_mins, pd.Series)
    assert len(target_maxes) == 1
    assert target_maxes["recall"] == max(recall_p, recall_q)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_1m_1sf_1cf(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    target = metrics.MetricFrame(
        metrics=skm.recall_score,
        y_true=transform_y_t(y_t),
        y_pred=transform_y_p(y_p),
        sensitive_features=g_2,
        control_features=g_3,
    )
    # Check on ci_quantiles
    assert target.ci_quantiles is None

    # Check on the indices properties
    assert isinstance(target.control_levels, list)
    assert target.control_levels == ["control_feature_0"]
    assert isinstance(target.sensitive_levels, list)
    assert target.sensitive_levels == ["sensitive_feature_0"]

    # Check we have correct return types
    assert isinstance(target.overall, pd.Series)
    assert isinstance(target.by_group, pd.Series)

    mask_f = g_2 == "f"
    mask_g = g_2 == "g"
    mask_k = g_3 == "kk"
    mask_m = g_3 == "m"

    # Check we have expected number of elements
    assert len(target.overall) == 2
    assert len(target.by_group) == 4

    recall_k = skm.recall_score(y_t[mask_k], y_p[mask_k])
    recall_m = skm.recall_score(y_t[mask_m], y_p[mask_m])
    assert target.overall["kk"] == recall_k
    assert target.overall["m"] == recall_m

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    assert target.by_group[("kk", "f")] == recall_k_f
    assert target.by_group[("kk", "g")] == recall_k_g
    assert target.by_group[("m", "f")] == recall_m_f
    assert target.by_group[("m", "g")] == recall_m_g

    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.Series)
    assert len(target_mins) == 2
    assert target_mins["kk"] == min(recall_k_arr)
    assert target_mins["m"] == min(recall_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.Series)
    assert len(target_maxs) == 2
    assert target_maxs["kk"] == max(recall_k_arr)
    assert target_maxs["m"] == max(recall_m_arr)


@pytest.mark.parametrize("transform_y_p", conversions_for_1d)
@pytest.mark.parametrize("transform_y_t", conversions_for_1d)
def test_1m_1sf_1cf_metric_dict(transform_y_t, transform_y_p):
    # If there are failures here, other, more specific tests should also fail
    target = metrics.MetricFrame(
        metrics={"recall": skm.recall_score},
        y_true=transform_y_t(y_t),
        y_pred=transform_y_p(y_p),
        sensitive_features=g_2,
        control_features=g_3,
    )
    # Check on ci_quantiles
    assert target.ci_quantiles is None

    # Check on the indices properties
    assert isinstance(target.control_levels, list)
    assert target.control_levels == ["control_feature_0"]
    assert isinstance(target.sensitive_levels, list)
    assert target.sensitive_levels == ["sensitive_feature_0"]

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    mask_f = g_2 == "f"
    mask_g = g_2 == "g"
    mask_k = g_3 == "kk"
    mask_m = g_3 == "m"

    # Check we have expected number of elements
    assert target.overall.shape == (2, 1)
    assert target.by_group.shape == (4, 1)

    recall_k = skm.recall_score(y_t[mask_k], y_p[mask_k])
    recall_m = skm.recall_score(y_t[mask_m], y_p[mask_m])
    assert target.overall["recall"]["kk"] == recall_k
    assert target.overall["recall"]["m"] == recall_m

    mask_k_f = np.logical_and(mask_k, mask_f)
    mask_k_g = np.logical_and(mask_k, mask_g)
    mask_m_f = np.logical_and(mask_m, mask_f)
    mask_m_g = np.logical_and(mask_m, mask_g)
    recall_k_f = skm.recall_score(y_t[mask_k_f], y_p[mask_k_f])
    recall_m_f = skm.recall_score(y_t[mask_m_f], y_p[mask_m_f])
    recall_k_g = skm.recall_score(y_t[mask_k_g], y_p[mask_k_g])
    recall_m_g = skm.recall_score(y_t[mask_m_g], y_p[mask_m_g])
    assert target.by_group["recall"][("kk", "f")] == recall_k_f
    assert target.by_group["recall"][("kk", "g")] == recall_k_g
    assert target.by_group["recall"][("m", "f")] == recall_m_f
    assert target.by_group["recall"][("m", "g")] == recall_m_g

    recall_k_arr = [recall_k_f, recall_k_g]
    recall_m_arr = [recall_m_f, recall_m_g]

    target_mins = target.group_min()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_mins.shape == (2, 1)
    assert target_mins["recall"]["kk"] == min(recall_k_arr)
    assert target_mins["recall"]["m"] == min(recall_m_arr)

    target_maxs = target.group_max()
    assert isinstance(target_mins, pd.DataFrame)
    assert target_maxs.shape == (2, 1)
    assert target_maxs["recall"]["kk"] == max(recall_k_arr)
    assert target_maxs["recall"]["m"] == max(recall_m_arr)


def test_1m_1_sf_sample_weights():
    """Check that sample weights are passed correctly to a single metric."""

    def multi_sp(y_t, y_p, p1, p2):
        """Metric to check passing of sample parameters.

        Verifies that p2 == y_t + y_p + p1 for all elements
        """
        assert len(y_t) == len(y_p)
        assert len(y_t) == len(p1)
        assert len(y_t) == len(p2)
        assert np.array_equal(p2, np.asarray(y_t) + np.asarray(y_p) + np.asarray(p1))
        return sum(p2)

    # Generate some random input data
    rng = np.random.default_rng(seed=42)
    param1 = rng.random(len(y_t))
    # Compute the expected sum
    param2 = s_w + y_p + param1

    # Note that we pass in the s_w array for y_true, to get
    # a little more variety in the results
    target = metrics.MetricFrame(
        metrics=multi_sp,
        y_true=s_w,
        y_pred=y_p,
        sensitive_features=g_1,
        sample_params={"p1": param1, "p2": param2},
    )
    # Check on ci_quantiles
    assert target.ci_quantiles is None

    # Sanity check types
    assert isinstance(target.overall, float)
    assert isinstance(target.by_group, pd.Series)
    assert target.by_group.shape == (2,)

    # Check the overall value
    assert target.overall == sum(param2)

    # Look at the by_group values for each subgroup identified by g_1
    for g in g_1:
        mask = g_1 == g
        assert target.by_group[g] == sum(param2[mask])


def test_2m_1sf_sample_weights():
    """Check that sample weights are passed correctly to two metrics."""

    def sp_is_sum(y_t, y_p, some_param_name):
        """Metric accepting a single sample parameter.

        Checks that the sample parameter stores the sum of
        y_t and y_p for all elements
        """
        assert len(y_t) == len(y_p)
        assert len(y_t) == len(some_param_name)
        assert np.array_equal(some_param_name, y_t + y_p)
        assert isinstance(some_param_name, np.ndarray)
        return sum(some_param_name)

    def multi_sp(y_t, y_p, some_param_name, some_other):
        """Metric accepting multiple sample parameters.

        Checks that
        some_other == y_t + y_p * some_param_name
        for all elements
        """
        assert len(y_t) == len(y_p)
        assert len(y_t) == len(some_param_name)
        assert len(y_t) == len(some_other)
        assert np.array_equal(some_other, y_t + y_p * some_param_name)
        assert isinstance(some_other, np.ndarray)
        return sum(some_other)

    # Give the metrics some unusual names
    m1 = r"! # \ | $"
    m2 = r"& % ^"
    metrics_dict = {m1: sp_is_sum, m2: multi_sp}

    # Generate some random inputs for multi_sp
    rng = np.random.default_rng(seed=6 * 9)
    param1 = rng.integers(low=0, high=100, size=len(y_t))

    # Generate the columns of expected values for the two metrics
    sums = y_t + s_w
    multis = y_t + s_w * param1

    # Set up the sample parameter dictionary
    # Note that although the metric functions have one parameter
    # with the same name, each one gets a different input array
    sample_params = {
        m1: {"some_param_name": sums},
        m2: {"some_param_name": param1, "some_other": multis},
    }

    # Compute the metrics. Note that we pass in the
    # s_w array for y_pred, in the interests of having some
    # more variety in the results
    target = metrics.MetricFrame(
        metrics=metrics_dict,
        y_true=y_t,
        y_pred=s_w,
        sensitive_features=g_2,
        sample_params=sample_params,
    )
    # Check on ci_quantiles
    assert target.ci_quantiles is None

    # Check we have correct return types
    assert isinstance(target.overall, pd.Series)
    assert isinstance(target.by_group, pd.DataFrame)

    # Check we have expected number of elements
    assert len(target.overall) == 2
    assert target.by_group.shape == (2, 2)

    # Check overall values
    assert target.overall[m1] == sum(sums)
    assert target.overall[m2] == sum(multis)

    # Check by group values for each subgroup identified by g_2
    for group in g_2:
        mask = g_2 == group
        assert target.by_group[m1][group] == sum(sums[mask])
        assert target.by_group[m2][group] == sum(multis[mask])


def test_duplicate_sf_names():
    groups = pd.DataFrame(np.stack([g_2, g_3], axis=1), columns=["A", "A"])
    msg = "Detected duplicate feature name: 'A'"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics.MetricFrame(
            metrics=skm.recall_score,
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=groups,
        )
    assert execInfo.value.args[0] == msg


def test_duplicate_cf_names():
    groups = pd.DataFrame(np.stack([g_2, g_3], axis=1), columns=["B", "B"])
    msg = "Detected duplicate feature name: 'B'"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics.MetricFrame(
            metrics=skm.recall_score,
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=g_4,
            control_features=groups,
        )
    assert execInfo.value.args[0] == msg


def test_duplicate_cf_sf_names():
    cf = pd.DataFrame(np.stack([g_2, g_3], axis=1), columns=["A", "B"])
    sf = {"B": g_1, "C": g_4}
    msg = "Detected duplicate feature name: 'B'"
    with pytest.raises(ValueError) as execInfo:
        _ = metrics.MetricFrame(
            metrics=skm.recall_score,
            y_true=y_t,
            y_pred=y_p,
            sensitive_features=sf,
            control_features=cf,
        )
    assert execInfo.value.args[0] == msg


def test_single_element_lists():
    mf = metrics.MetricFrame(
        metrics=skm.balanced_accuracy_score,
        y_true=[1],
        y_pred=[1],
        sensitive_features=[0],
    )
    assert mf.overall == 1
