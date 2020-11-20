# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import sklearn.metrics as skm

from ._extra_metrics import (
    true_positive_rate, true_negative_rate,
    false_positive_rate, false_negative_rate,
    selection_rate,
)
from . import make_derived_metric

METRICS_SPEC = [
    # base metrics from _extra_metrics
    (true_positive_rate, ["difference", "ratio"]),
    (true_negative_rate, ["difference", "ratio"]),
    (false_positive_rate, ["difference", "ratio"]),
    (false_negative_rate, ["difference", "ratio"]),
    (selection_rate, ["difference", "ratio"]),

    # base metrics from sklearn.metrics
    (skm.accuracy_score, ["difference", "ratio", "group_min"]),
    (skm.zero_one_loss, ["difference", "ratio", "group_max"]),
    (skm.balanced_accuracy_score, ["group_min"]),
    (skm.precision_score, ["group_min"]),
    (skm.recall_score, ["group_min"]),
    (skm.roc_auc_score, ["group_min"]),
    (skm.mean_absolute_error, ["group_max"]),
    (skm.mean_squared_error, ["group_max"]),
    (skm.r2_score, ["group_min"]),
    (skm.f1_score, ["group_min"]),
    (skm.log_loss, ["group_max"]),
]

_generated_metric_dict = dict()
for base_metric, variants in METRICS_SPEC:
    for variant in variants:
        name = "{0}_{1}".format(base_metric.__name__, variant)
        fn = make_derived_metric(metric=base_metric,
                                 transform=variant,
                                 sample_param_names=['sample_weight'])
        fn.__name__ = name
        _generated_metric_dict[name] = fn
