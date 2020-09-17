# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

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


def test_basic():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=gid)

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    # Check we have expected number of elements
    assert target.overall.shape == (1, 1)
    assert target.by_group.shape == (2, 1)

    expected_overall = skm.recall_score(y_t, y_p)
    assert target.overall['recall_score'][0] == expected_overall

    expected_A = skm.recall_score(y_t_A, y_p_A)
    assert target.by_group[('recall_score',)][('A',)] == expected_A
    expected_B = skm.recall_score(y_t_B, y_p_B)
    assert target.by_group[('recall_score',)][('B',)] == expected_B

    max_vals = target.group_max()
    assert isinstance(max_vals, pd.Series)
    assert len(max_vals) == 1
    assert max_vals[('recall_score',)] == max([expected_A, expected_B])

    min_vals = target.group_min()
    assert isinstance(min_vals, pd.Series)
    assert len(min_vals) == 1
    assert min_vals[('recall_score',)] == min([expected_A, expected_B])

    diffs = target.difference()
    assert isinstance(diffs, pd.Series)
    assert len(diffs) == 1
    assert diffs[('recall_score',)] == abs(expected_A - expected_B)


def test_basic_with_broadcast_arg():
    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   sensitive_features=gid,
                                   params={'pos_label': 0})

    # Check we have expected number of elements
    assert target.overall.shape == (1, 1)
    assert target.by_group.shape == (2, 1)

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

    # Check we have expected number of elements
    assert target.overall.shape == (1, 1)
    assert target.by_group.shape == (2, 1)

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

    # Check we have expected number of elements
    assert target.overall.shape == (1, 1)
    assert target.by_group.shape == (2, 1)

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

    # Check we have expected number of elements
    assert target.overall.shape == (1, 1)
    assert target.by_group.shape == (4, 1)

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

    expected = [expected_A_x, expected_A_y, expected_B_x, expected_B_y]
    max_vals = target.group_max()
    assert isinstance(max_vals, pd.Series)
    assert len(max_vals) == 1
    assert max_vals[('recall_score',)] == max(expected)


def test_single_conditional_feature():
    cf = ['x' if (x % 2) == 0 else 'y' for x in range(len(y_t))]
    cf_x_mask = [(x % 2) == 0 for x in range(len(y_t))]
    cf_y_mask = [(x % 2) == 1 for x in range(len(y_t))]

    target = metrics.GroupedMetric(skm.recall_score,
                                   y_t, y_p,
                                   conditional_features=[cf],
                                   sensitive_features=[gid])

    # Check we have expected number of elements
    assert target.overall.shape == (2, 1)
    assert target.by_group.shape == (2, 2)

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

    expected_x = [expected_A_x, expected_B_x]
    expected_y = [expected_A_y, expected_B_y]
    max_vals = target.group_max()
    assert isinstance(max_vals, pd.Series)
    assert len(max_vals) == 2
    assert max_vals[('recall_score', 'x')] == max(expected_x)
    assert max_vals[('recall_score', 'y')] == max(expected_y)


def test_two_metrics():
    fns = [skm.recall_score, skm.precision_score]
    sample_param_names = [['sample_weight'], ['sample_weight']]
    params = [{'sample_weight': wgt, 'pos_label': 0},
              {'sample_weight': wgt}]

    target = metrics.GroupedMetric(fns,
                                   y_t, y_p,
                                   sensitive_features=gid,
                                   sample_param_names=sample_param_names,
                                   params=params)

    # Check we have correct return types
    assert isinstance(target.overall, pd.DataFrame)
    assert isinstance(target.by_group, pd.DataFrame)

    # Check we have expected number of elements
    assert target.overall.shape == (1, 2)
    assert target.by_group.shape == (2, 2)

    recall_overall = skm.recall_score(y_t, y_p, sample_weight=wgt, pos_label=0)
    assert target.overall['recall_score'][0] == recall_overall
    precision_overall = skm.precision_score(y_t, y_p, sample_weight=wgt)
    assert target.overall['precision_score'][0] == precision_overall

    recall_A = skm.recall_score(y_t_A, y_p_A, sample_weight=wgt_A, pos_label=0)
    assert target.by_group[('recall_score',)][('A',)] == recall_A
    recall_B = skm.recall_score(y_t_B, y_p_B, sample_weight=wgt_B, pos_label=0)
    assert target.by_group[('recall_score',)][('B',)] == recall_B

    precision_A = skm.precision_score(y_t_A, y_p_A, sample_weight=wgt_A)
    assert target.by_group[('precision_score',)][('A',)] == precision_A
    precision_B = skm.precision_score(y_t_B, y_p_B, sample_weight=wgt_B)
    assert target.by_group[('precision_score',)][('B',)] == precision_B

    expected_recall = [recall_A, recall_B]
    expected_precision = [precision_A, precision_B]
    max_vals = target.group_max()
    assert isinstance(max_vals, pd.Series)
    assert len(max_vals) == 2
    assert max_vals[('recall_score',)] == max(expected_recall)
    assert max_vals[('precision_score',)] == max(expected_precision)


def test_two_metrics_two_sensitive_features():
    sf_2 = ['x' if (x % 2) == 0 else 'y' for x in range(len(y_t))]

    target = metrics.GroupedMetric([skm.recall_score, skm.precision_score],
                                   y_t, y_p,
                                   sensitive_features=[gid, sf_2])

    # Check we have expected number of elements
    assert target.overall.shape == (1, 2)
    assert target.by_group.shape == (4, 2)

    recall_overall = skm.recall_score(y_t, y_p)
    assert target.overall['recall_score'][0] == recall_overall
    precision_overall = skm.precision_score(y_t, y_p)
    assert target.overall['precision_score'][0] == precision_overall

    expected_recall = []
    expected_precision = []

    A_x_mask = [x % 2 == 0 for x in range(len(y_t_A))]
    A_y_mask = [x % 2 == 1 for x in range(len(y_t_A))]
    y_t_A_x = np.asarray(y_t_A)[A_x_mask]
    y_p_A_x = np.asarray(y_p_A)[A_x_mask]
    expected_A_x = skm.recall_score(y_t_A_x, y_p_A_x)
    assert target.by_group[('recall_score',)][('A', 'x')] == expected_A_x
    expected_recall.append(expected_A_x)
    expected_A_x = skm.precision_score(y_t_A_x, y_p_A_x)
    assert target.by_group[('precision_score',)][('A', 'x')] == expected_A_x
    expected_precision.append(expected_A_x)
    y_t_A_y = np.asarray(y_t_A)[A_y_mask]
    y_p_A_y = np.asarray(y_p_A)[A_y_mask]
    expected_A_y = skm.recall_score(y_t_A_y, y_p_A_y)
    assert target.by_group[('recall_score',)][('A', 'y')] == expected_A_y
    expected_recall.append(expected_A_y)
    expected_A_y = skm.precision_score(y_t_A_y, y_p_A_y)
    assert target.by_group[('precision_score',)][('A', 'y')] == expected_A_y
    expected_precision.append(expected_A_y)

    # Rely on B's being after the A's, but don't assume even numbers
    B_x_mask = [x % 2 == 0 for x in range(len(y_t_A), len(y_t))]
    B_y_mask = [x % 2 == 1 for x in range(len(y_t_A), len(y_t))]
    y_t_B_x = np.asarray(y_t_B)[B_x_mask]
    y_p_B_x = np.asarray(y_p_B)[B_x_mask]
    expected_B_x = skm.recall_score(y_t_B_x, y_p_B_x)
    assert target.by_group[('recall_score',)][('B', 'x')] == expected_B_x
    expected_recall.append(expected_B_x)
    expected_B_x = skm.precision_score(y_t_B_x, y_p_B_x)
    assert target.by_group[('precision_score',)][('B', 'x')] == expected_B_x
    expected_precision.append(expected_B_x)
    y_t_B_y = np.asarray(y_t_B)[B_y_mask]
    y_p_B_y = np.asarray(y_p_B)[B_y_mask]
    expected_B_y = skm.recall_score(y_t_B_y, y_p_B_y)
    assert target.by_group[('recall_score',)][('B', 'y')] == expected_B_y
    expected_recall.append(expected_B_y)
    expected_B_y = skm.precision_score(y_t_B_y, y_p_B_y)
    assert target.by_group[('precision_score',)][('B', 'y')] == expected_B_y
    expected_precision.append(expected_B_y)

    max_vals = target.group_max()
    assert isinstance(max_vals, pd.Series)
    assert len(max_vals) == 2
    assert max_vals[('recall_score',)] == max(expected_recall)
    assert max_vals[('precision_score',)] == max(expected_precision)

    min_vals = target.group_min()
    assert isinstance(min_vals, pd.Series)
    assert len(min_vals) == 2
    assert min_vals[('recall_score',)] == min(expected_recall)
    assert min_vals[('precision_score',)] == min(expected_precision)

    diffs = target.difference()
    assert isinstance(diffs, pd.Series)
    assert len(diffs) == 2
    assert diffs[('recall_score',)] == max(expected_recall) - min(expected_recall)
    assert diffs[('precision_score',)] == max(expected_precision) - min(expected_precision)



def test_two_metrics_single_conditional_feature():
    cf = ['x' if (x % 2) == 0 else 'y' for x in range(len(y_t))]
    cf_x_mask = [(x % 2) == 0 for x in range(len(y_t))]
    cf_y_mask = [(x % 2) == 1 for x in range(len(y_t))]

    fns = [skm.recall_score, skm.precision_score]
    sample_param_names = [['sample_weight'], ['sample_weight']]
    params = [{'sample_weight': wgt},
              {'sample_weight': wgt}]
    target = metrics.GroupedMetric(fns,
                                   y_t, y_p,
                                   conditional_features=[cf],
                                   sensitive_features=[gid],
                                   sample_param_names=sample_param_names,
                                   params=params)

    # Check we have expected number of elements
    assert target.overall.shape == (2, 2)
    assert target.by_group.shape == (2, 4)

    overall_x = skm.recall_score(y_t[cf_x_mask], y_p[cf_x_mask], sample_weight=wgt[cf_x_mask])
    assert target.overall['recall_score'][('x',)] == overall_x
    overall_y = skm.recall_score(y_t[cf_y_mask], y_p[cf_y_mask], sample_weight=wgt[cf_y_mask])
    assert target.overall['recall_score'][('y',)] == overall_y

    overall_x = skm.precision_score(y_t[cf_x_mask], y_p[cf_x_mask], sample_weight=wgt[cf_x_mask])
    assert target.overall['precision_score'][('x',)] == overall_x
    overall_y = skm.precision_score(y_t[cf_y_mask], y_p[cf_y_mask], sample_weight=wgt[cf_y_mask])
    assert target.overall['precision_score'][('y',)] == overall_y

    expected_recall_x = []
    expected_recall_y = []
    expected_precision_x = []
    expected_precision_y = []
    # This will look somewhat familiar from above.... mathematically
    # there's no difference between conditional and sensitive features
    # The difference is how they are inserted into the table
    A_x_mask = [x % 2 == 0 for x in range(len(y_t_A))]
    A_y_mask = [x % 2 == 1 for x in range(len(y_t_A))]
    y_t_A_x = np.asarray(y_t_A)[A_x_mask]
    y_p_A_x = np.asarray(y_p_A)[A_x_mask]
    wgt_A_x = np.asarray(wgt_A)[A_x_mask]
    expected_A_x = skm.recall_score(y_t_A_x, y_p_A_x, sample_weight=wgt_A_x)
    assert target.by_group[('recall_score', 'x')][('A',)] == expected_A_x
    expected_recall_x.append(expected_A_x)
    expected_A_x = skm.precision_score(y_t_A_x, y_p_A_x, sample_weight=wgt_A_x)
    assert target.by_group[('precision_score', 'x')][('A',)] == expected_A_x
    expected_precision_x.append(expected_A_x)
    y_t_A_y = np.asarray(y_t_A)[A_y_mask]
    y_p_A_y = np.asarray(y_p_A)[A_y_mask]
    wgt_A_y = np.asarray(wgt_A)[A_y_mask]
    expected_A_y = skm.recall_score(y_t_A_y, y_p_A_y, sample_weight=wgt_A_y)
    assert target.by_group[('recall_score', 'y')][('A',)] == expected_A_y
    expected_recall_y.append(expected_A_y)
    expected_A_y = skm.precision_score(y_t_A_y, y_p_A_y, sample_weight=wgt_A_y)
    assert target.by_group[('precision_score', 'y')][('A',)] == expected_A_y
    expected_precision_y.append(expected_A_y)

    # Rely on B's being after the A's, but don't assume even numbers
    B_x_mask = [x % 2 == 0 for x in range(len(y_t_A), len(y_t))]
    B_y_mask = [x % 2 == 1 for x in range(len(y_t_A), len(y_t))]
    y_t_B_x = np.asarray(y_t_B)[B_x_mask]
    y_p_B_x = np.asarray(y_p_B)[B_x_mask]
    wgt_B_x = np.asarray(wgt_B)[B_x_mask]
    expected_B_x = skm.recall_score(y_t_B_x, y_p_B_x, sample_weight=wgt_B_x)
    assert target.by_group[('recall_score', 'x')][('B',)] == expected_B_x
    expected_recall_x.append(expected_B_x)
    expected_B_x = skm.precision_score(y_t_B_x, y_p_B_x, sample_weight=wgt_B_x)
    assert target.by_group[('precision_score', 'x')][('B',)] == expected_B_x
    expected_precision_x.append(expected_B_x)
    y_t_B_y = np.asarray(y_t_B)[B_y_mask]
    y_p_B_y = np.asarray(y_p_B)[B_y_mask]
    wgt_B_y = np.asarray(wgt_B)[B_y_mask]
    expected_B_y = skm.recall_score(y_t_B_y, y_p_B_y, sample_weight=wgt_B_y)
    assert target.by_group[('recall_score', 'y')][('B',)] == expected_B_y
    expected_recall_y.append(expected_B_y)
    expected_B_y = skm.precision_score(y_t_B_y, y_p_B_y, sample_weight=wgt_B_y)
    assert target.by_group[('precision_score', 'y')][('B',)] == expected_B_y
    expected_precision_y.append(expected_B_y)

    max_vals = target.group_max()
    assert isinstance(max_vals, pd.Series)
    assert len(max_vals) == 4
    assert max_vals[('recall_score', 'x')] == max(expected_recall_x)
    assert max_vals[('recall_score', 'y')] == max(expected_recall_y)
    assert max_vals[('precision_score', 'x')] == max(expected_precision_x)
    assert max_vals[('precision_score', 'y')] == max(expected_precision_y)

    min_vals = target.group_min()
    assert isinstance(min_vals, pd.Series)
    assert len(min_vals) == 4
    assert min_vals[('recall_score', 'x')] == min(expected_recall_x)
    assert min_vals[('recall_score', 'y')] == min(expected_recall_y)
    assert min_vals[('precision_score', 'x')] == min(expected_precision_x)
    assert min_vals[('precision_score', 'y')] == min(expected_precision_y)

    diffs = target.difference()
    assert isinstance(diffs, pd.Series)
    assert len(diffs) == 4
    assert diffs[('recall_score', 'x')] == max(expected_recall_x) - min(expected_recall_x)
    assert diffs[('recall_score', 'y')] == max(expected_recall_y) - min(expected_recall_y)
    assert diffs[('precision_score', 'x')] == max(expected_precision_x) - min(expected_precision_x)
    assert diffs[('precision_score', 'y')] == max(expected_precision_y) - min(expected_precision_y)

