# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics
from test.unit.input_convertors import conversions_for_1d

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
    assert target.by_group['recall_score']['A'] == expected_A
    expected_B = skm.recall_score(y_t_B, y_p_B)
    assert target.by_group['recall_score']['B'] == expected_B
