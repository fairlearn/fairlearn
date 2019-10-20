# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm

from .metrics_engine import metric_by_group


def group_accuracy_score(y_true, y_pred, group_membership,
                         normalize=True,
                         sample_weight=None):

    def internal_acc_wrapper(y_true, y_pred, sample_weight=None):
        return skm.accuracy_score(y_true, y_pred,
                                  normalize,
                                  sample_weight)

    return metric_by_group(internal_acc_wrapper, y_true, y_pred, group_membership, sample_weight)


def group_confusion_matrix(y_true, y_pred, group_membership,
                           labels=None,
                           sample_weight=None):

    def internal_cm_wrapper(y_true, y_pred, sample_weight=None):
        return skm.confusion_matrix(y_true, y_pred,
                                    labels,
                                    sample_weight)

    return metric_by_group(internal_cm_wrapper, y_true, y_pred, group_membership, sample_weight)


def group_precision_score(y_true, y_pred, group_membership,
                          labels=None, pos_label=1, average='binary',
                          sample_weight=None):

    def internal_prec_wrapper(y_true, y_pred, sample_weight=None):
        return skm.precision_score(y_true, y_pred,
                                   labels=labels, pos_label=pos_label, average=average,
                                   sample_weight=sample_weight)

    return metric_by_group(internal_prec_wrapper, y_true, y_pred, group_membership, sample_weight)
