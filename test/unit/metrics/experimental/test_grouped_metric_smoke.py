# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

# ============================================================


y_t_A = [0, 1, 1, 1, 0, 0, 1, 1, 0]
y_p_A = [1, 1, 1, 0, 0, 0, 0, 1, 0]
wgt_A = [1, 2, 1, 2, 1, 2, 1, 3, 4]
As = np.full(len(y_t_A), 'A')
y_t_B = [0, 1, 1, 0, 0, 0, 0, 1]
y_p_B = [0, 1, 0, 1, 0, 1, 0, 1]
wgt_B = [2, 2, 1, 1, 3, 3, 4, 1]
Bs = np.full(len(y_t_B), 'B')

y_t = np.concatenate((y_t_A, y_t_B))
y_p = np.concatenate((y_p_A, y_p_B))
wgt = np.concatenate((wgt_A, wgt_B))
gid = np.concatenate((As, Bs))


def test_basic():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=gid)

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    expected_overall = skm.recall_score(y_t, y_p)
    assert target.overall['recall_score'][0] == expected_overall

    expected_A = skm.recall_score(y_t_A, y_p_A)
    assert target.by_group[('recall_score',)][('A',)] == expected_A
    expected_B = skm.recall_score(y_t_B, y_p_B)
    assert target.by_group[('recall_score',)][('B',)] == expected_B


def test_basic_with_broadcast_arg():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=gid,
                                   params={'pos_label': 0})

    expected_overall = skm.recall_score(y_t, y_p, pos_label=0)
    assert target.overall['recall_score'][0] == expected_overall

    expected_A = skm.recall_score(y_t_A, y_p_A, pos_label=0)
    assert target.by_group[('recall_score',)][('A',)] == expected_A
    expected_B = skm.recall_score(y_t_B, y_p_B, pos_label=0)
    assert target.by_group[('recall_score',)][('B',)] == expected_B


def test_basic_with_sample_arg():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=gid,
                                   sample_param_names=['sample_weight'],
                                   params={'sample_weight': wgt})

    expected_overall = skm.recall_score(y_t, y_p, sample_weight=wgt)
    assert target.overall['recall_score'][0] == expected_overall

    expected_A = skm.recall_score(y_t_A, y_p_A, sample_weight=wgt_A)
    assert target.by_group[('recall_score',)][('A',)] == expected_A
    expected_B = skm.recall_score(y_t_B, y_p_B, sample_weight=wgt_B)
    assert target.by_group[('recall_score',)][('B',)] == expected_B


def test_basic_with_broadcast_and_sample_arg():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=gid,
                                   sample_param_names=['sample_weight'],
                                   params={'sample_weight': wgt, 'pos_label': 0})

    expected_overall = skm.recall_score(y_t, y_p, sample_weight=wgt, pos_label=0)
    assert target.overall['recall_score'][0] == expected_overall

    expected_A = skm.recall_score(y_t_A, y_p_A, sample_weight=wgt_A, pos_label=0)
    assert target.by_group[('recall_score',)][('A',)] == expected_A
    expected_B = skm.recall_score(y_t_B, y_p_B, sample_weight=wgt_B, pos_label=0)
    assert target.by_group[('recall_score',)][('B',)] == expected_B


def test_two_sensitive_features():
    sf_2 = ['x' if (x % 2) == 0 else 'y' for x in range(len(y_t))]

    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=[gid, sf_2])

    expected_overall = skm.recall_score(y_t, y_p)
    assert target.overall['recall_score'][0] == expected_overall

    A_x_mask = [x % 2 == 0 for x in range(len(y_t_A))]
    A_y_mask = [x % 2 == 1 for x in range(len(y_t_A))]
    y_t_A_x = np.asarray(y_t_A)[A_x_mask]
    y_p_A_x = np.asarray(y_p_A)[A_x_mask]
    expected_A_x = skm.recall_score(y_t_A_x, y_p_A_x)
    assert target.by_group[('recall_score',)][('A', 'x')] == expected_A_x
    y_t_A_y = np.asarray(y_t_A)[A_y_mask]
    y_p_A_y = np.asarray(y_p_A)[A_y_mask]
    expected_A_y = skm.recall_score(y_t_A_y, y_p_A_y)
    assert target.by_group[('recall_score',)][('A', 'y')] == expected_A_y

    # Rely on B's being after the A's, but don't assume even numbers
    B_x_mask = [x % 2 == 0 for x in range(len(y_t_A), len(y_t))]
    B_y_mask = [x % 2 == 1 for x in range(len(y_t_A), len(y_t))]
    y_t_B_x = np.asarray(y_t_B)[B_x_mask]
    y_p_B_x = np.asarray(y_p_B)[B_x_mask]
    expected_B_x = skm.recall_score(y_t_B_x, y_p_B_x)
    assert target.by_group[('recall_score',)][('B', 'x')] == expected_B_x
    y_t_B_y = np.asarray(y_t_B)[B_y_mask]
    y_p_B_y = np.asarray(y_p_B)[B_y_mask]
    expected_B_y = skm.recall_score(y_t_B_y, y_p_B_y)
    assert target.by_group[('recall_score',)][('B', 'y')] == expected_B_y


def test_single_conditional_feature():
    cf = ['x' if (x % 2) == 0 else 'y' for x in range(len(y_t))]
    cf_x_mask = [(x % 2) == 0 for x in range(len(y_t))]
    cf_y_mask = [(x % 2) == 1 for x in range(len(y_t))]

    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   conditional_features=[cf],
                                   sensitive_features=[gid])

    overall_x = skm.recall_score(y_t[cf_x_mask], y_p[cf_x_mask])
    assert target.overall['recall_score'][('x',)] == overall_x
    overall_y = skm.recall_score(y_t[cf_y_mask], y_p[cf_y_mask])
    assert target.overall['recall_score'][('y',)] == overall_y

    # This will look somewhat familiar from above.... mathematically
    # there's no difference between conditional and sensitive features
    # The difference is how they are inserted into the table
    A_x_mask = [x % 2 == 0 for x in range(len(y_t_A))]
    A_y_mask = [x % 2 == 1 for x in range(len(y_t_A))]
    y_t_A_x = np.asarray(y_t_A)[A_x_mask]
    y_p_A_x = np.asarray(y_p_A)[A_x_mask]
    expected_A_x = skm.recall_score(y_t_A_x, y_p_A_x)
    assert target.by_group[('recall_score', 'x')][('A',)] == expected_A_x
    y_t_A_y = np.asarray(y_t_A)[A_y_mask]
    y_p_A_y = np.asarray(y_p_A)[A_y_mask]
    expected_A_y = skm.recall_score(y_t_A_y, y_p_A_y)
    assert target.by_group[('recall_score', 'y')][('A',)] == expected_A_y

    # Rely on B's being after the A's, but don't assume even numbers
    B_x_mask = [x % 2 == 0 for x in range(len(y_t_A), len(y_t))]
    B_y_mask = [x % 2 == 1 for x in range(len(y_t_A), len(y_t))]
    y_t_B_x = np.asarray(y_t_B)[B_x_mask]
    y_p_B_x = np.asarray(y_p_B)[B_x_mask]
    expected_B_x = skm.recall_score(y_t_B_x, y_p_B_x)
    assert target.by_group[('recall_score', 'x')][('B',)] == expected_B_x
    y_t_B_y = np.asarray(y_t_B)[B_y_mask]
    y_p_B_y = np.asarray(y_p_B)[B_y_mask]
    expected_B_y = skm.recall_score(y_t_B_y, y_p_B_y)
    assert target.by_group[('recall_score', 'y')][('B',)] == expected_B_y
