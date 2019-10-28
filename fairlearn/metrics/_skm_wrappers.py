# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm

from ._metrics_engine import metric_by_group


def group_accuracy_score(y_true, y_pred, group_membership, *,
                         normalize=True,
                         sample_weight=None):
    """A wrapper around the :any:`sklearn.metrics.accuracy_score` routine.
    The arguments remain the same, with `group_membership` added.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """

    def internal_acc_wrapper(y_true, y_pred, sample_weight=None):
        return skm.accuracy_score(y_true, y_pred,
                                  normalize,
                                  sample_weight)

    return metric_by_group(internal_acc_wrapper, y_true, y_pred, group_membership, sample_weight)


def group_confusion_matrix(y_true, y_pred, group_membership, *,
                           labels=None,
                           sample_weight=None):
    """A wrapper around the :any:`sklearn.metrics.confusion_matrix` routine.
    The arguments remain the same, with `group_membership` added.
    However, the only positional arguments supported are `y_true`,
    `y_pred` and `group_membership`.
    All others must be specified by name.
    """

    def internal_cm_wrapper(y_true, y_pred, sample_weight=None):
        return skm.confusion_matrix(y_true, y_pred,
                                    labels,
                                    sample_weight)

    return metric_by_group(internal_cm_wrapper, y_true, y_pred, group_membership, sample_weight)


def group_precision_score(y_true, y_pred, group_membership, *,
                          labels=None, pos_label=1, average='binary',
                          sample_weight=None):
    """A wrapper around the :any:`sklearn.metrics.precision_score` routine.
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
    """A wrapper around the :any:`sklearn.metrics.recall_score` routine.
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
    """A wrapper around the :any:`sklearn.metrics.roc_auc_score` routine.
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
    """A wrapper around the :any:`sklearn.metrics.zero_one_loss` routine.
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
    """A wrapper around the :any:`sklearn.metrics.mean_squared_error` routine.
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
