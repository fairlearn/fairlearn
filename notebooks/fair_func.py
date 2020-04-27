import logging
import numpy
import numpy as np
import pandas as pd
from datetime import datetime
from operator import itemgetter
from zipfile import ZipFile
from io import BytesIO
from urllib.request import urlopen
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from math import *
from sklearn import ensemble
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


class DataTransformer:

    CATEGORICAL_VARIABLES = {"Country": ["EE", "ES", "FI", "SK"],
                             "CreditScoreEeMini": [0.0, 600.0, 700.0, 800.0, 900.0, 1000.0],
                             "CreditScoreEsEquifaxRisk": ["A", "AA", "AAA", "B", "C", "D"],
                             "CreditScoreEsMicroL": ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"],
                             "CreditScoreFiAsiakasTietoRiskGrade": ["RL0", "RL1", "RL2", "RL3", "RL4", "RL5"],
                             "Education": [1.0, 2.0, 3.0, 4.0, 5.0],
                             "EmploymentDurationCurrentEmployer": ["MoreThan5Years", "TrialPeriod", "UpTo1Year",
                                                                   "UpTo2Years", "UpTo3Years", "UpTo4Years",
                                                                   "UpTo5Years", "Retiree", "Other"],
                             "EmploymentStatus": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                             "Gender": [0.0, 1.0, 2.0],
                             "HomeOwnershipType": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                             "LanguageCode": [1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 21, 22],
                             "MaritalStatus": [1.0, 2.0, 3.0, 4.0, 5.0],
                             "MonthlyPaymentDay": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                   20, 21, 22, 23, 24, 25, 26, 27, 28],
                             "NewCreditCustomer": [False, True],
                             "OccupationArea": [-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                                13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                             "Rating": ["A", "AA", "B", "C", "D", "E", "F", "HR"],
                             "UseOfLoan": [0, 1, 2, 3, 4, 5, 6, 7, 8, 101, 102, 104, 106, 107, 108, 110],
                             "VerificationType": [1.0, 2.0, 3.0, 4.0],
                             "NrOfDependants": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10Plus"],
                             "WorkExperience": ["10To15Years", "15To25Years", "2To5Years", "5To10Years",
                                                "LessThan2Years", "MoreThan25Years"]}


    NUMERIC_VARIABLES = ["Age", "AppliedAmount", "DebtToIncome", "ExpectedLoss", "LiabilitiesTotal", "FreeCash",
                         "IncomeFromChildSupport", "IncomeFromFamilyAllowance", "IncomeFromLeavePay",
                         "IncomeFromPension", "IncomeFromPrincipalEmployer", "IncomeFromSocialWelfare", "IncomeOther",
                         "IncomeTotal", "Interest", "LoanDuration", "LossGivenDefault", "MonthlyPayment",
                         "ProbabilityOfDefault", "NrOfDependantslessthan3", "WrExLess10", "WrExLess5", "Tenant", "Default"]

    PREDICTOR_VARIABLES = sorted(list(CATEGORICAL_VARIABLES.keys())) + NUMERIC_VARIABLES

    @classmethod
    def assign_categories(cls, column):
        """

        :param column: all columns of df
        :return: all categorical columns
        """
        return column.astype("category", categories=cls.CATEGORICAL_VARIABLES[column.name])

    @classmethod
    def transform(cls, data):
        """

        :param data: data frame
        :return: dummies values after OHE
        """
        data = data[cls.PREDICTOR_VARIABLES]
        data[cls.NUMERIC_VARIABLES] = data[cls.NUMERIC_VARIABLES].astype("float64")
        ordered_categorical_keys = sorted(list(cls.CATEGORICAL_VARIABLES.keys()))
        data[ordered_categorical_keys] = data[ordered_categorical_keys].apply(cls.assign_categories)
        return pd.get_dummies(data)





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
    tn_ww, fp_ww, fn_ww, tp_ww = confusion_matrix(y_true, Y_pred_binary_ww).ravel() #y_true, y_pred
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

    tnr_ww = tn_ww/(tn_ww + fp_ww)
    tnr_wow = tn_wow/(tn_wow + fp_wow)


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
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true[X_test[protected_label]==pval], Y_pred_binary[X_test[protected_label]==pval]).ravel()
    tn_up, fp_up, fn_up, tp_up = confusion_matrix(y_true[X_test[protected_label]==upval], Y_pred_binary[X_test[protected_label]==upval]).ravel()


    roc_p = roc_auc_score(y_true[X_test[protected_label]==pval], y_pred_prob[X_test[protected_label]==pval])
    roc_up = roc_auc_score(y_true[X_test[protected_label]==upval], y_pred_prob[X_test[protected_label]==upval])


    ps_p=average_precision_score(y_true[X_test[protected_label]==pval], y_pred_prob[X_test[protected_label]==pval])
    ps_up=average_precision_score(y_true[X_test[protected_label]==upval], y_pred_prob[X_test[protected_label]==upval])


    EOpp_p = tp_p/(tp_p+fn_p) #protected and unprotected groups have equal FNR
    EOpp_up = tp_up/(tp_up+fn_up) #protected and unprotected groups have equal FNR


    EOdds_p = (fp_p / (fp_p+tn_p)) + (tp_p / (tp_p+fn_p)) #equal TPR + FPR
    EOdds_up = (fp_up / (fp_p+tn_up)) + (tp_up / (tp_p+fn_up)) #Equal TPR + FPR


    prec_p = tp_p / (tp_p+fp_p)
    prec_up = tp_up / (tp_up+fp_up)

    demo_parity_p = (tp_p + fp_p) / (tn_p + fp_p + fn_p + tp_p)
    demo_parity_up = (tp_up + fp_up) / (tn_up + fp_up + fn_up + tp_up)


    fpr_p = fp_p / (fp_p + tn_p)
    fpr_up = fp_up / (fp_up + tn_up)
    tpr_p = tp_p / (tp_p + fn_p)
    tpr_up = tp_up / (tp_up + fn_up)
    AOD=0.5*((fpr_up-fpr_p)+(tpr_up-tpr_p))


    p_eq_p = fpr_p #protected and unprotected groups have equal FPR
    p_eq_up = fpr_up #protected and unprotected groups have equal FPR


    TE_p = fn_p/fp_p
    TE_up = fn_up/fp_up


    pp_p = tp_p / (tp_p + fp_p) #both protected and unprotected groups have equal PPV
    pp_up = tp_up / (tp_up + fp_up) ##both protected and unprotected groups have equal PPV


    cost_p = (fp_p*700) + (fn_p*300)
    cost_up = (fp_up*700) + (fn_up*300)

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

        #predicted = predicted_proba.apply(lambda x: 0 if x > threshold else 1)

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
        sensitivity = tp /(tp + fn)
        fnr = fn/(fn + tp)



        cost = (fp*700) + (fn*300)

        return EOpp, EOdds, demo_parity, AOD, p_eq, TE, pp, TNR, precision, FPR, f1, roc, sensitivity, fnr, fp, fn, cost #, sensitivity, fnr,

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



        results_p = metrics_eval_thresh(y_true[X_test[protected_label]==pval], predicted_proba[X_test[protected_label]==pval], t1)

        results_up = metrics_eval_thresh(y_true[X_test[protected_label]==upval], predicted_proba[X_test[protected_label]==upval], t1)



        EOpp_p, EOdds_p, demo_parity_p, AOD_p, p_eq_p, TE_p, pp_p, TNR_p, precision_p, FPR_p, f1_p, roc_p, sensitivity_p, fnr_p, fp_p, fn_p, cost_p = results_p
        EOpp_up, EOdds_up, demo_parity_up, AOD_up, p_eq_up, TE_up, pp_up, TNR_up, precision_up, FPR_up, f1_up, roc_up, sensitivity_up, fnr_up, fp_up, fn_up, cost_up = results_up






        EOpp.append(EOpp_up-EOpp_p)
        EOdds.append(EOdds_up - EOdds_p)
        demo_parity.append(demo_parity_up - demo_parity_p)

        AOD.append(AOD_up-AOD_p)
        p_eq.append(p_eq_up-p_eq_p)
        TE.append(TE_up-TE_p)
        pp.append(pp_up-pp_p)



        TNR.append (TNR_up-TNR_p)
        precision.append (precision_up-precision_p)
        FPR.append (FPR_up-FPR_p)
        f1.append (f1_up - f1_p)
        #sensitivity.append(roc_up-roc_p)
        roc.append(roc_up-roc_p)
        sensitivity.append(sensitivity_up-sensitivity_p)
        fnr.append(fnr_up-fnr_p)
        fp.append (fp_up - fp_p)
        fn.append (fn_up-fn_p)

        cost.append(cost_up+cost_p)



        #print (t1)
        #print (EOpp_up-EOpp_p)
        #print (EOdds_up - EOdds_p)
        #print (demo_parity_up - demo_parity_p)
        #print (cost_up-cost_p)
        #print ("--------------")

    return EOpp, EOdds, demo_parity, AOD, p_eq, TE, pp, TNR, precision, FPR, f1, roc, sensitivity, fnr, fp, fn, cost






def on_diff_threshold(X_test, y_true, protected_label, predicted_proba, pval, upval):



    """
    Fairness, cost and accuracy  metrics for a model to compare priviliged and unpriviliged across independent
    for threshold for both the groups



     Parameters
    ----------

    :param X_test: Xtest data
    :param y_true: Actual binary outcome
    :param protected_label: Sensitive feature
    :param predicted_proba: predicted probabilities
    :param pval: Priviliged value of protected label
    :param upval: Unpriviliged value of protected label
    :return: two tables one for priviliged group and another for unpriviliged gropu with cost, metrics
            EOpp, EOdds, demo_parity, FNR, TE, ppv, cost


    Examples
    --------
    t_p, t_up, metric, Cost = on_diff_threshold(X_test1, y_test, choice, y_pred_prob_wow, pval, upval)

    """


    def metrics_eval_thresh(y_true, predicted_proba, threshold):

        #predicted = predicted_proba.apply(lambda x: 0 if x > threshold else 1)

        predicted = [0 if (i > threshold) else 1 for i in predicted_proba]

        tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
        EOpp = tp/(tp+fn)
        EOdds = fp / (fp+tn)
        demo_parity = (tp + fp) / (tn + fp + fn + tp)
        #FPR = EOdds
        #TPR = EOpp
        FNR = fn / (fn+tp)
        TE = fn/fp
        ppv = tp / (tp+fp)
        cost = (fp*700) + (fn*300)
        #aor = FPR - TPR
        return EOpp, EOdds, demo_parity, FNR, TE, ppv, cost



    thresholds = np.arange(.3, .9, 0.01)

    #df_p = X_test[X_test[protected_label]==pval]
    #df_up = X_test[X_test[protected_label]==upval]

    t_p = []
    t_up = []
    metrics = []
    Cost = []

    for t0 in thresholds:
        for t1 in thresholds:

            results_p = metrics_eval_thresh(y_true[X_test[protected_label]==pval], predicted_proba[X_test[protected_label]==pval], t0)

            results_up = metrics_eval_thresh(y_true[X_test[protected_label]==upval], predicted_proba[X_test[protected_label]==upval], t1)

            ######

            #½ [(FPR_{S=unprivileged}−FPR_{S=privileged}) + (TPR_{S=privileged}−TPR_{S=unprivileged})]


            ####

            EOpp_p, EOdds_p, demo_parity_p, FNR_p, TE_p, ppv_p, cost_p = results_p
            EOpp_up, EOdds_up, demo_parity_up, FNR_up, TE_up, ppv_up, cost_up = results_up



            if round(EOpp_p, 4) == round(EOpp_up, 4):
                #print (t0, t1)
                #print (EOpp_p,EOpp_up)
                #print ("--------------")
                t_p.append(t0)
                t_up.append(t1)
                metrics.append("Eq of Opp")
                Cost.append(cost_p+cost_up)


            if round(EOdds_p, 4) == round(EOdds_up, 4):
                #print (t0, t1)
                #print (EOdds_p,EOdds_up)
                #print ("--------------")
                t_p.append(t0)
                t_up.append(t1)
                metrics.append("Eq of Odds")
                Cost.append(cost_p+cost_up)

            if round(demo_parity_p, 4) == round(demo_parity_up, 4):
                #print (t0, t1)
                #print (demo_parity_p,demo_parity_up)
                #print ("--------------")
                t_p.append(t0)
                t_up.append(t1)
                metrics.append("Demographic Parity")
                Cost.append(cost_p+cost_up)

            if round(FNR_p, 4) == round(FNR_up, 4):
                #print (t0, t1)
                #print (demo_parity_p,demo_parity_up)
                #print ("--------------")
                t_p.append(t0)
                t_up.append(t1)
                metrics.append("False Negative Rate")
                Cost.append(cost_p+cost_up)

            if round(TE_p, 3) == round(TE_up, 3):
                #print (t0, t1)
                #print (demo_parity_p,demo_parity_up)
                #print ("--------------")
                t_p.append(t0)
                t_up.append(t1)
                metrics.append("Treatment Equality")
                Cost.append(cost_p+cost_up)

            if round(ppv_p, 3) == round(ppv_up, 3):
                #print (t0, t1)
                #print (demo_parity_p,demo_parity_up)
                #print ("--------------")
                t_p.append(t0)
                t_up.append(t1)
                metrics.append("Positive Pred Value")
                Cost.append(cost_p+cost_up)



    return t_p, t_up, metrics, Cost


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
    
    
    EOpp_p = tp_p/(tp_p+fn_p) #protected and unprotected groups have equal FNR
    EOpp_up = tp_up/(tp_up+fn_up) #protected and unprotected groups have equal FNR


    EOdds_p = (fp_p / (fp_p+tn_p)) + (tp_p / (tp_p+fn_p)) #equal TPR + FPR
    EOdds_up = (fp_up / (fp_p+tn_up)) + (tp_up / (tp_p+fn_up)) #Equal TPR + FPR


    prec_p = tp_p / (tp_p+fp_p)
    prec_up = tp_up / (tp_up+fp_up)

    demo_parity_p = (tp_p + fp_p) / (tn_p + fp_p + fn_p + tp_p)
    demo_parity_up = (tp_up + fp_up) / (tn_up + fp_up + fn_up + tp_up)


    fpr_p = fp_p / (fp_p + tn_p)
    fpr_up = fp_up / (fp_up + tn_up)
    tpr_p = tp_p / (tp_p + fn_p)
    tpr_up = tp_up / (tp_up + fn_up)
    AOD=0.5*((fpr_up-fpr_p)+(tpr_up-tpr_p))


    p_eq_p = fpr_p #protected and unprotected groups have equal FPR
    p_eq_up = fpr_up #protected and unprotected groups have equal FPR


    pp_p = tp_p / (tp_p + fp_p) #both protected and unprotected groups have equal PPV
    pp_up = tp_up / (tp_up + fp_up) ##both protected and unprotected groups have equal PPV
    
    
    cost_p = (fp_p*700) + (fn_p*300)
    cost_up = (fp_up*700) + (fn_up*300)
    
    
    
    
    return (abs(EOpp_up-EOpp_p), abs(EOdds_up-EOdds_p), abs(prec_up-prec_p), abs(demo_parity_up-demo_parity_p), abs(AOD), abs(p_eq_up-p_eq_p),  abs(pp_up-pp_p), abs(tpr_up-tpr_p), abs((cost_up+cost_p)/10000000))



def RMSE(y_pred, y_test):
    rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / len(y_test) )
    return rmse# test error



from numpy import inf
def mape(y_pred, y_true):
    
    
    df=pd.DataFrame(np.abs((y_pred - y_true)/y_true))
    df=df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    xx=np.mean(df)*100
    mape = xx.values[0]
    
    return mape




def thetha(y_test, y_pred_prob, X_test, protected_label, pval = 0, DECISION_THRESHOLD = 0.5, demote = False):
    upval = int(not pval) #Unprivileged
    
    pred_prob = y_pred_prob.copy() #probabilities of getting output as 0 or favourable (high probability high case of 0 or fav)
    s = X_test[protected_label]
    
    flip_candidates = np.ones_like(pred_prob).astype(bool) \
            if demote else s == upval #unprivileged group
    
    thetha = np.arange(0.01, 0.99, 0.01)
    
    ROC=[]
    cnt = 0
    for t1 in thetha:
        under_theta_index = np.where(
            (np.abs(pred_prob - 0.5) < t1) & flip_candidates & (pred_prob<0.5))
        
        pred_prob[under_theta_index] = 1-pred_prob[under_theta_index] #flipping the probabilities
        
        rocdata=pred_prob.copy()
        rocbin=np.where(rocdata > 0.5, 0, 1)
        
        tn_up, fp_up, fn_up, tp_up = confusion_matrix(y_test[s==upval], rocbin[s==upval]).ravel()
        tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_test[s==pval], rocbin[s==pval]).ravel()
        results = acfmetrics(tn_up, fp_up, fn_up, tp_up, tn_p, fp_p, fn_p, tp_p)
        
        #print(results)
        
        ROC.append(results)
        
    
    return ROC
