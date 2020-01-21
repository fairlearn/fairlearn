# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm
from math import sqrt
from ._metrics_engine import make_group_metric, metric_by_group


def group_accuracy_score(y_true, y_pred, group_membership, *,
                         normalize=True,
                         sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.accuracy_score` routine.

group_confusion_matrix = make_group_metric(skm.confusion_matrix)
"""A grouped wrapper around the :py:func:`sklearn.metrics.confusion_matrix` routine."""

group_precision_score = make_group_metric(skm.precision_score)
"""A grouped wrapper around the :py:func:`sklearn.metrics.precision_score` routine
"""

group_recall_score = make_group_metric(skm.recall_score)
"""A grouped wrapper around the :py:func:`sklearn.metrics.recall_score` routine
"""

def group_confusion_matrix(y_true, y_pred, group_membership, *,
                           labels=None,
                           sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.confusion_matrix` routine.

group_zero_one_loss = make_group_metric(skm.zero_one_loss)
"""A grouped wrapper around the :py:func:`sklearn.metrics.zero_one_loss` routine
"""


def group_precision_score(y_true, y_pred, group_membership, *,
                          labels=None, pos_label=1, average='binary',
                          sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.precision_score` routine.

    The arguments remain the same, with `group_membership` added.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """
    def internal_prec_wrapper(y_true, y_pred, sample_weight=None):
        return skm.precision_score(y_true, y_pred,
                                   labels=labels, pos_label=pos_label, average=average,
                                   sample_weight=sample_weight)

    return metric_by_group(internal_prec_wrapper, y_true, y_pred, group_membership, sample_weight)


def group_recall_score(y_true, y_pred, group_membership, *,
                       labels=None, pos_label=1, average='binary',
                       sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.recall_score` routine.

    The arguments remain the same, with `group_membership` added.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """
    def internal_recall_wrapper(y_true, y_pred, sample_weight=None):
        return skm.recall_score(y_true, y_pred,
                                labels=labels, pos_label=pos_label, average=average,
                                sample_weight=sample_weight)

    return metric_by_group(internal_recall_wrapper,
                           y_true, y_pred, group_membership, sample_weight)


def group_roc_auc_score(y_true, y_pred, group_membership, *,
                        average='macro', max_fpr=None,
                        sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.roc_auc_score` routine.

    The arguments remain the same, with `group_membership` added.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """
    def internal_ras_wrapper(y_true, y_pred, sample_weight=None):
        return skm.roc_auc_score(y_true, y_pred,
                                 average=average, max_fpr=max_fpr,
                                 sample_weight=sample_weight)

    return metric_by_group(internal_ras_wrapper,
                           y_true, y_pred, group_membership, sample_weight)


def group_zero_one_loss(y_true, y_pred, group_membership, *,
                        normalize=True,
                        sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.zero_one_loss` routine.

    The arguments remain the same, with `group_membership` added.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """
    def internal_zol_wrapper(y_true, y_pred, sample_weight=None):
        return skm.zero_one_loss(y_true, y_pred,
                                 normalize=normalize,
                                 sample_weight=sample_weight)

    return metric_by_group(internal_zol_wrapper, y_true, y_pred, group_membership, sample_weight)

# --------------------------------------------------------------------------------------


def group_mean_squared_error(y_true, y_pred, group_membership, *,
                             multioutput='uniform_average',
                             sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.mean_squared_error` routine.

    The arguments remain the same, with `group_membership` added.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """
    def internal_mse_wrapper(y_true, y_pred, sample_weight=None):
        return skm.mean_squared_error(y_true, y_pred,
                                      multioutput=multioutput,
                                      sample_weight=sample_weight)

    return metric_by_group(internal_mse_wrapper,
                           y_true, y_pred, group_membership, sample_weight=sample_weight)


def group_root_mean_squared_error(y_true, y_pred, group_membership, *,
                                  multioutput='uniform_average',
                                  sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.mean_squared_error` routine.

    The arguments remain the same, with `group_membership` added.
    The result is then square rooted.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """
    def internal_rmse_wrapper(y_true, y_pred, sample_weight=None):
        return sqrt(skm.mean_squared_error(y_true, y_pred,
                                           multioutput=multioutput,
                                           sample_weight=sample_weight))

    return metric_by_group(internal_rmse_wrapper,
                           y_true, y_pred, group_membership, sample_weight=sample_weight)


def group_r2_score(y_true, y_pred, group_membership, *,
                   multioutput='uniform_average',
                   sample_weight=None):
    """Wrap the :py:func:`sklearn.metrics.r2_score` routine.

group_max_error = make_group_metric(skm.max_error)
"""A grouped wrapper around the :py:func:`sklearn.metrics.max_error` routine
"""

    return metric_by_group(internal_r2_wrapper,
                           y_true, y_pred, group_membership, sample_weight=sample_weight)


group_max_error = make_group_metric(skm.max_error)
"""A grouped wrapper around the :py:func:`sklearn.metrics.max_error` routine
"""

group_mean_absolute_error = make_group_metric(skm.mean_absolute_error)
"""A grouped wrapper around the :py:func:`sklearn.metrics.mean_absolute_error` routine
"""

group_mean_squared_log_error = make_group_metric(skm.mean_squared_log_error)
"""A grouped wrapper around the :py:func:`sklearn.metrics.mean_squared_log_error` routine
"""

group_median_absolute_error = make_group_metric(skm.median_absolute_error)
"""A grouped wrapper around the :py:func:`sklearn.metrics.median_absolute_error` routine
"""
