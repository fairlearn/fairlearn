# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Metrics for measuring fairness."""

import numpy as np
from sklearn.metrics import roc_curve
import math
from scipy.stats import t

from ._base_metrics import false_positive_rate, selection_rate, true_positive_rate
from ._metric_frame import MetricFrame


def demographic_parity_difference(
    y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None
) -> float:
    """Calculate the demographic parity difference.

    The demographic parity difference is defined as the difference
    between the largest and the smallest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The demographic parity difference of 0 means that all groups have the same selection rate.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.difference`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The demographic parity difference
    """
    sel_rate = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight},
    )
    result = sel_rate.difference(method=method)
    return result


def demographic_parity_ratio(
    y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None
) -> float:
    """Calculate the demographic parity ratio.

    The demographic parity ratio is defined as the ratio
    between the smallest and the largest group-level selection rate,
    :math:`E[h(X) | A=a]`, across all values :math:`a` of the sensitive feature(s).
    The demographic parity ratio of 1 means that all groups have the same selection rate.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The demographic parity ratio
    """
    sel_rate = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params={"sample_weight": sample_weight},
    )
    result = sel_rate.ratio(method=method)
    return result


def equalized_odds_difference(
    y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None
) -> float:
    """Calculate the equalized odds difference.

    The greater of two metrics: `true_positive_rate_difference` and
    `false_positive_rate_difference`. The former is the difference between the
    largest and smallest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds difference of 0 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.difference`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The equalized odds difference
    """
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    return max(eo.difference(method=method))


def equalized_odds_ratio(
    y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None
) -> float:
    """Calculate the equalized odds ratio.

    The smaller of two metrics: `true_positive_rate_ratio` and
    `false_positive_rate_ratio`. The former is the ratio between the
    smallest and largest of :math:`P[h(X)=1 | A=a, Y=1]`, across all values :math:`a`
    of the sensitive feature(s). The latter is defined similarly, but for
    :math:`P[h(X)=1 | A=a, Y=0]`.
    The equalized odds ratio of 1 means that all groups have the same
    true positive, true negative, false positive, and false negative rates.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels :math:`h(X)` returned by the classifier.

    sensitive_features :
        The sensitive features over which demographic parity should be assessed

    method : str
        How to compute the differences. See :func:`fairlearn.metrics.MetricFrame.ratio`
        for details.

    sample_weight : array-like
        The sample weights

    Returns
    -------
    float
        The equalized odds ratio
    """
    eo = _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight)

    return min(eo.ratio(method=method))


def _get_eo_frame(y_true, y_pred, sensitive_features, sample_weight) -> MetricFrame:
    fns = {"tpr": true_positive_rate, "fpr": false_positive_rate}
    sw_dict = {"sample_weight": sample_weight}
    sp = {"tpr": sw_dict, "fpr": sw_dict}
    eo = MetricFrame(
        metrics=fns,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sp,
    )
    return eo

def significance_level(
    y_true_1, y_pred_1, y_true_2, y_pred_2
) -> float:
    """Calculate the significance level of the difference between two metrics.

    The significance level is the probability of obtaining a difference between
    the two metrics that is as large as or larger than the observed difference
    between the metrics, assuming that the two metrics are equal.

    Read more in the :ref:`User Guide <disparity_metrics>`.

    Parameters
    ----------
    y_true_1 : array-like
        Ground truth (correct) labels for the first set of predictions.

    y_pred_1 : array-like
        Predicted labels :math:`h(X)` returned by the classifier for the first set of predictions.

    y_true_2 : array-like
        Ground truth (correct) labels for the second set of predictions.

    y_pred_2 : array-like
        Predicted labels :math:`h(X)` returned by the classifier for the second set of predictions.

    Returns
    -------
    float
        The significance level of the difference between the two metrics
    """
    control_case_1 = _get_control_case(y_true_1, y_pred_1)
    control_case_2 = _get_control_case(y_true_2, y_pred_2)

    nR = len(control_case_1['controls'])
    mR = len(control_case_1['cases'])

    nS = len(control_case_2['controls'])
    mS = len(control_case_2['cases'])

    VR = _delong_placements(control_case_1)
    VS = _delong_placements(control_case_2)

    SRX = sum((x - VR['theta']) ** 2 for x in VR['X']) / (mR - 1)
    SSX = sum((x - VS['theta']) ** 2 for x in VS['X']) / (mS - 1)

    SRY = sum((VR["Y"] - VR["theta"]) ** 2) / (nR-1)
    SSY = sum((VS["Y"] - VS["theta"]) ** 2) / (nS-1)

    SR = SRX/mR + SRY/nR
    SS = SSX/mS + SSY/nS

    ntotR = nR + mR
    ntotS = nS + mS
    SSR = math.sqrt((SR) + (SS))
    
    t_ = (VR["theta"] - VS["theta"]) / SSR
    df = ((SR) + (SS))**2 / (((SR)**2 / (ntotR-1)) + ((SS)**2 / (ntotS -1 )))
    p_value = 2 * t.sf(abs(t_), df)
    
    return p_value

def _get_control_case(y_true, y_pred):
    controls = []
    cases = []
    for i in range(len(y_true)):
        if y_true[i] == 0:
            controls.append(y_pred[i])
        else:
            cases.append(y_pred[i])
    control_case = {'controls': controls, 'cases': cases}

    fpr, tpr, _ = roc_curve(y_true[:], y_pred[:])
    if np.median(fpr) > np.median(tpr):
        control_case['direction'] = ">"
    else:
        control_case['direction'] = "<"

    return control_case

def _delong_placements(control_case):
    cases = control_case["cases"]
    controls = control_case["controls"]
    direction = control_case["direction"]
    
    m = len(cases)
    n = len(controls)
    L = m + n

    if direction == ">":
        cases = [-x for x in cases]
        controls = [-x for x in controls]
        
    Z = []
    labels = []

    for i in range(m):
        Z.append((i, cases[i]))
        labels.append(True)
        
    for j in range(n):
        Z.append((m+j, controls[j]))
        labels.append(False)

    Z = sorted(Z, key = lambda x: x[1])

    XY = np.zeros(L)
    X_inds, Y_inds = [], []
    
    m,n,i = 0,0,0
    
    while i < L:
        X_inds.clear()
        Y_inds.clear()
        mdupl, ndupl = 0, 0
        while True:
            j = Z[i][0]
            if labels[j]:
                mdupl += 1
                X_inds.append(j)
            else:
                ndupl += 1
                Y_inds.append(j)
            if i == L - 1:
                break
            if Z[i][1] != Z[i+1][1]:
                break
            i += 1
        for k in range(mdupl):
            XY[X_inds[k]] = n + ndupl/2.0
        for k in range(ndupl):
            XY[Y_inds[k]] = m + mdupl/2.0
        n += ndupl
        m += mdupl
        i += 1
    sum_ = 0.0
    X, Y = [], []
    for i in range(L):
        if labels[i]:
            sum_ += XY[i]
            X.append(XY[i]/n)
        else:
            Y.append(1 - XY[i]/m)
    
    theta = sum_ / m / n
    return {'X': X, 'Y': Y, 'theta': theta}