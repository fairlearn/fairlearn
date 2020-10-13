# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.metrics import MetricsFrame
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

    gm = MetricsFrame(selection_rate, y_t, y_p, sensitive_features=g_1)

    assert actual == gm.difference(method='minmax')['selection_rate']


def test_demographic_parity_difference_weighted():
    actual = demographic_parity_difference(y_t, y_p,
                                           sensitive_features=g_1,
                                           sample_weight=s_w)

    gm = MetricsFrame(selection_rate, y_t, y_p,
                      sensitive_features=g_1,
                      sample_params={'sample_weight': s_w})

    assert actual == gm.difference(method='minmax')['selection_rate']


def test_demographic_parity_ratio():
    actual = demographic_parity_ratio(y_t, y_p, sensitive_features=g_1)

    gm = MetricsFrame(selection_rate, y_t, y_p, sensitive_features=g_1)

    assert actual == gm.ratio(method='minmax')['selection_rate']


def test_demographic_parity_ratio_weighted():
    actual = demographic_parity_ratio(y_t, y_p,
                                      sensitive_features=g_1,
                                      sample_weight=s_w)

    gm = MetricsFrame(selection_rate, y_t, y_p,
                      sensitive_features=g_1,
                      sample_params={'sample_weight': s_w})

    assert actual == gm.ratio(method='minmax')['selection_rate']


def test_equalized_odds_difference():
    actual = equalized_odds_difference(y_t, y_p, sensitive_features=g_1)

    metrics = {'tpr': true_positive_rate, 'fpr': false_positive_rate}
    gm = MetricsFrame(metrics, y_t, y_p, sensitive_features=g_1)

    diffs = gm.difference(method='minmax')
    assert actual == diffs.max()


def test_equalized_odds_difference_weighted():
    actual = equalized_odds_difference(y_t, y_p, sensitive_features=g_1, sample_weight=s_w)

    metrics = {'tpr': true_positive_rate, 'fpr': false_positive_rate}
    sw = {'sample_weight': s_w}
    sp = {'tpr': sw, 'fpr': sw}
    gm = MetricsFrame(metrics, y_t, y_p, sensitive_features=g_1, sample_params=sp)

    diffs = gm.difference(method='minmax')
    assert actual == diffs.max()


def test_equalized_odds_ratio():
    actual = equalized_odds_ratio(y_t, y_p, sensitive_features=g_1)

    metrics = {'tpr': true_positive_rate, 'fpr': false_positive_rate}
    gm = MetricsFrame(metrics, y_t, y_p, sensitive_features=g_1)

    ratios = gm.ratio(method='minmax')
    assert actual == ratios.min()


def test_equalized_odds_ratio_weighted():
    actual = equalized_odds_ratio(y_t, y_p, sensitive_features=g_1, sample_weight=s_w)

    metrics = {'tpr': true_positive_rate, 'fpr': false_positive_rate}
    sw = {'sample_weight': s_w}
    sp = {'tpr': sw, 'fpr': sw}
    gm = MetricsFrame(metrics, y_t, y_p, sensitive_features=g_1, sample_params=sp)

    ratios = gm.ratio(method='minmax')
    assert actual == ratios.min()
