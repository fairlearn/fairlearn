# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.metrics import GroupedMetric
from fairlearn.metrics import selection_rate, true_positive_rate, false_positive_rate
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio
)

from .data_for_test import y_t, y_p, s_w, g_1


def test_demographic_parity_difference():
    actual = demographic_parity_difference(y_t, y_p, sensitive_features=g_1)

    gm = GroupedMetric(selection_rate, y_t, y_p, sensitive_features=g_1)

    assert actual == gm.difference(method='minmax')['selection_rate']


def test_demographic_parity_difference_weighted():
    actual = demographic_parity_difference(y_t, y_p,
                                           sensitive_features=g_1,
                                           sample_weight=s_w)

    gm = GroupedMetric(selection_rate, y_t, y_p,
                       sensitive_features=g_1,
                       sample_params={'sample_weight': s_w})

    assert actual == gm.difference(method='minmax')['selection_rate']
