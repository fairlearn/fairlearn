# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics
from .utils import _get_raw_GroupedMetric


y_t = np.asarray(
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0]
)

y_p = np.asarray(
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
        1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
)

s_w = np.asarray(
    [1, 2, 3, 1, 2, 1, 2, 3, 1, 1, 1, 2, 3, 3, 1, 2, 1, 1, 1, 2, 3,
        2, 2, 1, 2, 1, 3, 1, 3, 1, 4, 1, 2, 1, 3, 2, 3, 3, 4, 1, 4, 1]
)


def _group_gen(x, runs, groups):
    # For producing arrays such as ['a', 'a', 'b', 'b', 'a', 'a', ... ]
    assert runs >= len(groups)
    return groups[(x//runs) % len(groups)]


g_1 = np.asarray([_group_gen(x, 2, ['a', 'b']) for x in range(len(y_t))])
g_2 = np.asarray([_group_gen(x, 3, ['f', 'g']) for x in range(len(y_t))])
g_3 = np.asarray([_group_gen(x, 4, ['k', 'm']) for x in range(len(y_t))])
g_4 = np.asarray([_group_gen(x, 4, ['p', 'q']) for x in range(len(y_t))])


def test_1m_0cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score, None, None)
    result = target._compute_overall(func_dict, y_t, y_p, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    expected = skm.recall_score(y_t, y_p)
    assert result['recall_score']['overall'] == expected


def test_1m_0cf_wgt():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score,
                                          ['sample_weight'],
                                          {'sample_weight': s_w})
    
    result = target._compute_overall(func_dict, y_t, y_p, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 1)
    expected = skm.recall_score(y_t, y_p, sample_weight=s_w)
    assert result['recall_score']['overall'] == expected
