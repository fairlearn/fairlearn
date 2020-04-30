import numpy
import numpy as np
import pandas as pd
from numpy import inf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def gini(actual, pred):
    """
    :param actual: actual values
    :param pred: predicted probablities
    :return: gini scores
    """
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(actual, pred):
    """
    :param actual: actual values
    :param pred: predicted probablities
    :return: normalized gini scores
    """
    return gini(actual, pred) / gini(actual, actual)


def model_metrics(y_true, y_pred_prob_ww, y_pred_prob_wow, Y_pred_binary_ww, Y_pred_binary_wow, X_test):
    """
    Model accuracy metrics for models with sample weights and without sample weights

    Parameters
    ----------

    :param y_true: Actual binary outcome
    :param y_pred_prob_ww: predicted probabilities with weights
    :param y_pred_prob_wow: predicted probabilities without weights
    :param Y_pred_binary_ww: predicted binary outcome with weights
    :param Y_pred_binary_wow: predicted binary outcome without weights
    :param X_test: Xtest data [not being used here]
    :return: roc, gini, avg precision, precision, sensitivity, tnr, fnr, f1, cost

    Examples
    --------
    model_perf=[model_metrics(y_test, y_pred_prob_ww, y_pred_prob_wow,
                          y_pred_ww, y_pred_wow, X_test1)]

    """
    tn_ww, fp_ww, fn_ww, tp_ww = confusion_matrix(y_true, Y_pred_binary_ww).ravel()
    tn_wow, fp_wow, fn_wow, tp_wow = confusion_matrix(y_true, Y_pred_binary_wow).ravel()
    roc_ww = roc_auc_score(y_true, y_pred_prob_ww)
    roc_wow = roc_auc_score(y_true, y_pred_prob_wow)
    gini_ww = gini_normalized(y_true, y_pred_prob_ww)
    gini_wow = gini_normalized(y_true, y_pred_prob_wow)
    ps_ww = average_precision_score(y_true, Y_pred_binary_ww)
    ps_wow = average_precision_score(y_true, Y_pred_binary_wow)
    prec_ww = tp_ww / (tp_ww + fp_ww)
    prec_wow = tp_wow / (tp_wow + fp_wow)
    sensitivity_ww = tp_ww/(tp_ww+fn_ww)
    sensitivity_wow = tp_wow/(tp_wow+fn_wow)
    fnr_ww = fn_ww/(fn_ww+tp_ww)
    fnr_wow = fn_wow/(fn_wow+tp_wow)
    f1_ww = (2*tp_ww)/((2*tp_ww)+fp_ww+fn_ww)
    f1_wow = (2*tp_wow)/((2*tp_wow)+fp_wow+fn_wow)
    cost_ww = (fp_ww*700) + (fn_ww*300)
    cost_wow = (fp_wow*700) + (fn_wow*300)
    return roc_ww, gini_ww, ps_ww, prec_ww, sensitivity_ww, fnr_ww, f1_ww, cost_ww, roc_wow, gini_wow, ps_wow,  prec_wow, sensitivity_wow, fnr_wow, f1_wow, cost_wow


def perf_metrics(y_true, y_pred_prob, Y_pred_binary, X_test, protected_label, pval, upval):
    """
    Fairness performance metrics for a model to compare priviliged and unpriviliged groups of a protected variable

    Parameters
    ----------

    :param y_true: Actual binary outcome
    :param y_pred_prob: predicted probabilities
    :param Y_pred_binary: predicted binary outcome
    :param X_test: Xtest data
    :param protected_label: Sensitive feature
    :param pval: Priviliged value of protected label
    :param upval: Unpriviliged value of protected label
    :return: roc, avg precision, Eq of Opportunity, Eq Odds, Precision, Demographic Parity, Avg Odds Diff,
            Predictive equality, Treatment Eq, predictive parity, Cost

    Examples
    --------
    ww=[perf_metrics(y_test, y_pred_prob_ww, y_pred_ww, X_test1, choice, pval, upval)]
    wow=[perf_metrics(y_test, y_pred_prob_wow, y_pred_wow, X_test1, choice, pval, upval)]

    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true[X_test[protected_label] == pval], Y_pred_binary[X_test[protected_label] == pval]).ravel()
    tn_up, fp_up, fn_up, tp_up = confusion_matrix(y_true[X_test[protected_label] == upval], Y_pred_binary[X_test[protected_label] == upval]).ravel()
    roc_p = roc_auc_score(y_true[X_test[protected_label] == pval], y_pred_prob[X_test[protected_label] == pval])
    roc_up = roc_auc_score(y_true[X_test[protected_label] == upval], y_pred_prob[X_test[protected_label] == upval])
    ps_p = average_precision_score(y_true[X_test[protected_label] == pval], y_pred_prob[X_test[protected_label] == pval])
    ps_up = average_precision_score(y_true[X_test[protected_label] == upval], y_pred_prob[X_test[protected_label] == upval])
    EOpp_p = tp_p/(tp_p+fn_p)
    EOpp_up = tp_up/(tp_up+fn_up)
    EOdds_p = (fp_p / (fp_p+tn_p)) + (tp_p / (tp_p+fn_p))
    EOdds_up = (fp_up / (fp_p+tn_up)) + (tp_up / (tp_p+fn_up))
    prec_p = tp_p / (tp_p+fp_p)
    prec_up = tp_up / (tp_up+fp_up)
    demo_parity_p = (tp_p + fp_p) / (tn_p + fp_p + fn_p + tp_p)
    demo_parity_up = (tp_up + fp_up) / (tn_up + fp_up + fn_up + tp_up)
    fpr_p = fp_p / (fp_p + tn_p)
    fpr_up = fp_up / (fp_up + tn_up)
    tpr_p = tp_p / (tp_p + fn_p)
    tpr_up = tp_up / (tp_up + fn_up)
    AOD = 0.5*((fpr_up-fpr_p)+(tpr_up-tpr_p))
    p_eq_p = fpr_p
    p_eq_up = fpr_up
    TE_p = fn_p/fp_p
    TE_up = fn_up/fp_up
    pp_p = tp_p / (tp_p + fp_p)
    pp_up = tp_up / (tp_up + fp_up)
    cost_p = (fp_p * 700) + (fn_p * 300)
    cost_up = (fp_up * 700) + (fn_up * 300)
    return abs(roc_up-roc_p), abs(ps_up-ps_p), abs(EOpp_up-EOpp_p), abs(EOdds_up-EOdds_p), abs(prec_up-prec_p), abs(demo_parity_up-demo_parity_p), AOD, abs(p_eq_up-p_eq_p), abs(TE_up-TE_p), abs(pp_up-pp_p), cost_up-cost_p, cost_up+cost_p

def on_same_threshold(X_test, y_true, protected_label, predicted_proba, pval, upval):
    """
    Fairness, cost and accuracy  metrics for a model to compare priviliged and unpriviliged across common
    for threshold for both the groups

    Parameters
    ----------

    :param X_test: Xtest data
    :param y_true: Actual binary outcome
    :param protected_label: Sensitive feature
    :param predicted_proba: predicted probabilities
    :param pval: Priviliged value of protected label
    :param upval: Unpriviliged value of protected label
    :return: Eq of Opportunity, Eq of Odds, Demographic Parity, Avg Odds Difference, Predictive equality,
            Treatment Equality, Predictive parity, True Negative Rates, Precision, False Positive Rates
            f1, AUC, Sensitivity, False negative rates, False Positive, False Negative, Cost

    Examples
    --------
    EOpp, EOdds, demo_parity, AOD, p_eq, TE, pp, TNR, precision, FPR, f1, roc, sensitivity, fnr, fp, fn, cost =
    on_same_threshold(X_test1, y_test, choice, y_pred_prob_wow, pval, upval)

    """
    def metrics_eval_thresh(y_true, predicted_proba, threshold):
        predicted = [0 if (i > threshold) else 1 for i in predicted_proba]
        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
        fpr = fp / (fp + tn)
        tpr = tp / (tp + fn)
        EOpp = tp/(tp+fn)
        EOdds = (fp / (fp+tn)) + (tp / (tp+fn))
        demo_parity = (tp + fp) / (tn + fp + fn + tp)
        AOD = (fpr+tpr)/2
        p_eq = fpr
        TE = fn/fp
        pp = tp / (tp + fp)
        TNR = tn / (tn+fp)
        precision = tp / (tp+fp)
        FPR = fp / (fp+tn)
        f1 = (2*tp) / (2*tp + fp + fn)
        roc = roc_auc_score(y_true, predicted)
        sensitivity = tp / (tp + fn)
        fnr = fn/(fn + tp)
        cost = (fp*700) + (fn*300)
        return EOpp, EOdds, demo_parity, AOD, p_eq, TE, pp, TNR, precision, FPR, f1, roc, sensitivity, fnr, fp, fn, cost
    
    thresholds = np.arange(0, 1.01, 0.01)
    EOpp = []
    EOdds = []
    demo_parity = []
    AOD = []
    p_eq = []
    TE = []
    pp = []
    TNR = []
    precision = []
    FPR = []
    f1 = []
    roc = []
    sensitivity = []
    fnr = []
    fp = []
    fn = []
    cost = []
    for t1 in thresholds:
        results_p = metrics_eval_thresh(y_true[X_test[protected_label] == pval], predicted_proba[X_test[protected_label] == pval], t1)
        results_up = metrics_eval_thresh(y_true[X_test[protected_label] == upval], predicted_proba[X_test[protected_label] == upval], t1)
        EOpp_p, EOdds_p, demo_parity_p, AOD_p, p_eq_p, TE_p, pp_p, TNR_p, precision_p, FPR_p, f1_p, roc_p, sensitivity_p, fnr_p, fp_p, fn_p, cost_p = results_p
        EOpp_up, EOdds_up, demo_parity_up, AOD_up, p_eq_up, TE_up, pp_up, TNR_up, precision_up, FPR_up, f1_up, roc_up, sensitivity_up, fnr_up, fp_up, fn_up, cost_up = results_up
        EOpp.append(EOpp_up-EOpp_p)
        EOdds.append(EOdds_up - EOdds_p)
        demo_parity.append(demo_parity_up - demo_parity_p)
        AOD.append(AOD_up-AOD_p)
        p_eq.append(p_eq_up-p_eq_p)
        TE.append(TE_up-TE_p)
        pp.append(pp_up-pp_p)
        TNR.append(TNR_up-TNR_p)
        precision.append(precision_up-precision_p)
        FPR.append(FPR_up-FPR_p)
        f1.append(f1_up - f1_p)
        roc.append(roc_up-roc_p)
        sensitivity.append(sensitivity_up-sensitivity_p)
        fnr.append(fnr_up-fnr_p)
        fp.append(fp_up - fp_p)
        fn.append(fn_up-fn_p)
        cost.append(cost_up+cost_p)
    return EOpp, EOdds, demo_parity, AOD, p_eq, TE, pp, TNR, precision, FPR, f1, roc, sensitivity, fnr, fp, fn, cost








def acfmetrics(tn_up, fp_up, fn_up, tp_up, tn_p, fp_p, fn_p, tp_p):
    """

    Fairness metrics for a model to compare priviliged and unpriviliged given confusion matrix for each group

    Parameters
    ----------

    :param tn_up: TN Unpriviliged
    :param fp_up: FP Unpriviliged
    :param fn_up: FN Unpriviliged
    :param tp_up: TP Unpriviliged
    :param tn_p: TN Priviliged
    :param fp_p: FP Priviliged
    :param fn_p: FN Priviliged
    :param tp_p: TP Priviliged
    :return: Eq of Opportunity, Eq of Odds, Demographic Parity, Avg Odds Difference, Predictive equality, Predictive parity, TPR, Cost

    Examples
    --------
    ACFmodel = acfmetrics(tn_up, fp_up, fn_up, tp_up, tn_p, fp_p, fn_p, tp_p)

    """
    EOpp_p = tp_p / (tp_p + fn_p)
    EOpp_up = tp_up / (tp_up + fn_up)
    EOdds_p = (fp_p / (fp_p + tn_p)) + (tp_p / (tp_p + fn_p))
    EOdds_up = (fp_up / (fp_p + tn_up)) + (tp_up / (tp_p + fn_up))
    prec_p = tp_p / (tp_p + fp_p)
    prec_up = tp_up / (tp_up + fp_up)
    demo_parity_p = (tp_p + fp_p) / (tn_p + fp_p + fn_p + tp_p)
    demo_parity_up = (tp_up + fp_up) / (tn_up + fp_up + fn_up + tp_up)
    fpr_p = fp_p / (fp_p + tn_p)
    fpr_up = fp_up / (fp_up + tn_up)
    tpr_p = tp_p / (tp_p + fn_p)
    tpr_up = tp_up / (tp_up + fn_up)
    AOD = 0.5 * ((fpr_up - fpr_p) + (tpr_up - tpr_p))
    p_eq_p = fpr_p
    p_eq_up = fpr_up
    pp_p = tp_p / (tp_p + fp_p)
    pp_up = tp_up / (tp_up + fp_up)
    cost_p = (fp_p * 700) + (fn_p * 300)
    cost_up = (fp_up * 700) + (fn_up * 300)
    return (abs(EOpp_up-EOpp_p), abs(EOdds_up-EOdds_p), abs(prec_up-prec_p), abs(demo_parity_up-demo_parity_p), abs(AOD), abs(p_eq_up-p_eq_p),  abs(pp_up-pp_p), abs(tpr_up-tpr_p), abs((cost_up+cost_p)/10000000))


def RMSE(y_pred, y_test):
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return rmse


def mape(y_pred, y_true):
    df = pd.DataFrame(np.abs((y_pred - y_true) / y_true))
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    xx = np.mean(df) * 100
    mape = xx.values[0]
    return mape