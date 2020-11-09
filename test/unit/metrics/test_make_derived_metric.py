# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import numpy as np
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics


# ============================================================

y_t_A = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1]
y_p_A = [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]
wgt_A = [1, 2, 1, 2, 1, 2, 1, 3, 4, 1, 2, 1, 2, 1, 2, 1, 2, 5]
As = np.full(len(y_t_A), 'A')
y_t_B = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]
y_p_B = [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
wgt_B = [2, 2, 1, 1, 3, 3, 4, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 1]
Bs = np.full(len(y_t_B), 'B')

y_t = np.concatenate((y_t_A, y_t_B))
y_p = np.concatenate((y_p_A, y_p_B))
wgt = np.concatenate((wgt_A, wgt_B))
gid = np.concatenate((As, Bs))


def test_derived_difference_between_groups():
    my_fn = metrics.make_derived_metric(metric=skm.accuracy_score,
                                        transform='difference',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.accuracy_score,
                                  y_t, y_p,
                                  sensitive_features=gid)

    actual = my_fn(y_t, y_p, sensitive_features=gid, method='between_groups')
    assert actual == grouped.difference(method='between_groups')


def test_derived_difference_default_is_between_groups():
    my_fn = metrics.make_derived_metric(metric=skm.accuracy_score,
                                        transform='difference',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.accuracy_score,
                                  y_t, y_p,
                                  sensitive_features=gid)

    actual = my_fn(y_t, y_p, sensitive_features=gid)
    assert actual == grouped.difference()


def test_derived_difference_to_overall():
    my_fn = metrics.make_derived_metric(metric=skm.accuracy_score,
                                        transform='difference',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.accuracy_score,
                                  y_t, y_p,
                                  sensitive_features=gid)

    actual = my_fn(y_t, y_p, sensitive_features=gid, method='to_overall')
    assert actual == grouped.difference(method='to_overall')


def test_derived_difference_broadcast_arg():
    my_beta = 0.6
    my_fn = metrics.make_derived_metric(metric=skm.fbeta_score,
                                        transform='difference',
                                        sample_param_names=['sample_weight'])

    my_fbeta = functools.partial(skm.fbeta_score, beta=my_beta)
    my_fbeta.__name__ = "my_fbeta"
    grouped = metrics.MetricFrame(my_fbeta,
                                  y_t, y_p,
                                  sensitive_features=gid)

    actual = my_fn(y_t, y_p, sensitive_features=gid,
                   beta=my_beta, method='between_groups')
    assert actual == grouped.difference(method='between_groups')


def test_derived_difference_sample_arg():
    my_fbeta = functools.partial(skm.fbeta_score, beta=0.6)
    my_fbeta.__name__ = "my_fbeta"
    my_fn = metrics.make_derived_metric(metric=my_fbeta,
                                        transform='difference',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(my_fbeta,
                                  y_t, y_p,
                                  sensitive_features=gid,
                                  sample_params={'sample_weight': wgt})
    actual = my_fn(y_t, y_p, sensitive_features=gid,
                   sample_weight=wgt, method='between_groups')
    assert actual == grouped.difference(method='between_groups')


def test_derived_difference_both_arg_types():
    my_beta = 0.5
    my_fn = metrics.make_derived_metric(metric=skm.fbeta_score,
                                        transform='difference',
                                        sample_param_names=['sample_weight'])

    my_fbeta = functools.partial(skm.fbeta_score, beta=my_beta)
    my_fbeta.__name__ = "my_fbeta"
    grouped = metrics.MetricFrame(my_fbeta,
                                  y_t, y_p,
                                  sensitive_features=gid,
                                  sample_params={'sample_weight': wgt})

    actual = my_fn(y_t, y_p,
                   sensitive_features=gid,
                   beta=my_beta,
                   sample_weight=wgt,
                   method='between_groups')
    assert actual == grouped.difference(method='between_groups')


def test_derived_difference_both_arg_types_default_sample_param_names():
    my_beta = 0.5
    my_fn = metrics.make_derived_metric(metric=skm.fbeta_score,
                                        transform='difference')

    my_fbeta = functools.partial(skm.fbeta_score, beta=my_beta)
    my_fbeta.__name__ = "my_fbeta"
    grouped = metrics.MetricFrame(my_fbeta,
                                  y_t, y_p,
                                  sensitive_features=gid,
                                  sample_params={'sample_weight': wgt})

    actual = my_fn(y_t, y_p,
                   sensitive_features=gid,
                   beta=my_beta,
                   sample_weight=wgt)
    assert actual == grouped.difference()


def test_derived_ratio_between_groups():
    my_fn = metrics.make_derived_metric(metric=skm.precision_score,
                                        transform='ratio',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.precision_score,
                                  y_t, y_p,
                                  sensitive_features=gid)
    actual = my_fn(y_t, y_p, sensitive_features=gid, method='between_groups')
    assert actual == grouped.ratio(method='between_groups')


def test_derived_ratio_default_is_between_groups():
    my_fn = metrics.make_derived_metric(metric=skm.precision_score,
                                        transform='ratio',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.precision_score,
                                  y_t, y_p,
                                  sensitive_features=gid)
    actual = my_fn(y_t, y_p, sensitive_features=gid)
    assert actual == grouped.ratio()


def test_derived_ratio_to_overall():
    my_fn = metrics.make_derived_metric(metric=skm.precision_score,
                                        transform='ratio',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.precision_score,
                                  y_t, y_p,
                                  sensitive_features=gid)
    actual = my_fn(y_t, y_p, sensitive_features=gid, method='to_overall')
    assert actual == grouped.ratio(method='to_overall')


def test_group_min():
    my_fn = metrics.make_derived_metric(metric=skm.precision_score,
                                        transform='group_min',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.precision_score,
                                  y_t, y_p,
                                  sensitive_features=gid)
    actual = my_fn(y_t, y_p, sensitive_features=gid)
    assert actual == grouped.group_min()


def test_group_max():
    my_fn = metrics.make_derived_metric(metric=skm.precision_score,
                                        transform='group_max',
                                        sample_param_names=['sample_weight'])

    grouped = metrics.MetricFrame(skm.precision_score,
                                  y_t, y_p,
                                  sensitive_features=gid)
    actual = my_fn(y_t, y_p, sensitive_features=gid)
    assert actual == grouped.group_max()


def test_noncallable_rejected():
    expected = "Supplied metric object must be callable"
    with pytest.raises(ValueError) as context:
        _ = metrics.make_derived_metric(metric="random", transform='group_max')
    assert context.value.args[0] == expected


def test_function_method_arg_rejected():

    def bad_fn(y_p, y_t, method):
        print(method)
        return skm.accuracy_score(y_p, y_t)

    expected = "Callables which accept a 'method' argument" \
        " may not be passed to make_derived_metric()." \
        " Please use functools.partial()"
    with pytest.raises(ValueError) as context:
        _ = metrics.make_derived_metric(metric=bad_fn, transform='group_max')
    assert context.value.args[0] == expected


def test_bad_transform_rejected():
    expected = "Transform must be one of ['difference', 'group_min', 'group_max', 'ratio']"

    with pytest.raises(ValueError) as context:
        _ = metrics.make_derived_metric(metric=skm.accuracy_score,
                                        transform="something rotten")
    assert context.value.args[0] == expected
