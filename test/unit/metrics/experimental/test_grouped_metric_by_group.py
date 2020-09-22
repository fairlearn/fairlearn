# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import sklearn.metrics as skm

from .data_for_test import y_t, y_p, s_w, g_1, g_2
from .utils import _get_raw_GroupedMetric


def test_1m_1sf_0cf():
    target = _get_raw_GroupedMetric()
    func_dict = target._process_functions(skm.recall_score, None, None)
    sf_list = target._process_features("SF", g_1, len(y_t))

    result = target._compute_by_group(func_dict, y_t, y_p, sf_list, None)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1)
    assert np.array_equal(result.index.names, ['SF 0'])
    mask_a = (g_1 == 'a')
    mask_b = (g_1 == 'b')
    recall_a = skm.recall_score(y_t[mask_a], y_p[mask_a])
    recall_b = skm.recall_score(y_t[mask_b], y_p[mask_b])
    assert result['recall_score'][('a',)] == recall_a
    assert result['recall_score'][('b',)] == recall_b
