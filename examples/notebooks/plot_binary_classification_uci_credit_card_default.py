"""
==============================================================
Binary Classification with the UCI Credit-card Default Dataset
==============================================================
"""
print(__doc__)

# %%
# Mitigating disparities in accuracy as measured by equalized-odds difference
#
# Contents
# --------
#
# 1. `What is covered`_
# 2. `Introduction`_
# 3. `The UCI credit card default dataset`_
# 4. `Using a fairness unaware model`_
# 5. `Mitigating equalized odds difference with Postprocessing`_
# 6. `Mitigating equalized odds difference with GridSearch`_
#
# What is covered
# ---------------
#
# * **Domain:**
#
#   * Finance (loan decisions). The data is semi-synthetic to create a simple
#     example of disparity in accuracy.
#
# * **ML task:**
#
#   * Binary classification.
#
# * **Fairness tasks:**
#
#   * Assessment of unfairness using Fairlearn metrics and Fairlearn dashboard.
#   * Mitigation of unfairness using Fairlearn mitigation algorithms.
#
# * **Performance metrics:**
#
#   * Area under ROC curve.
#   * Balanced accuracy.
#
# * **Fairness metrics:**
#
#   * Equalized-odds difference.
#
# * **Mitigation algorithms:**
#
#   * :class:`fairlearn.reductions.GridSearch`
#   * :class:`fairlearn.postprocessing.ThresholdOptimizer`
#
# Introduction
# ------------
#
# In this example, we emulate the problem of accuracy disparities arising in
# loan decisions. Specifically, we consider scenarios where algorithmic tools
# are trained on historic data and their predictions about loan applicants
# are used for making decisions about applicants. See
# `here <https://www.nytimes.com/2019/11/10/business/Apple-credit-card-investigation.html>`_
# for an example involving sex-based discrimination for credit limit
# decisions.
#
# We use the
# `UCI dataset <https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients>`_
# on credit-card defaults in 2005 in Taiwan. For the sake of this exercise,
# we modify the original UCI dataset: we introduce a synthetic feature that
# has a strong predictive power for female clients, but is uninformative for
# male applicants. We fit a variety of models for predicting the default of a
# client. We show that a fairness-unaware training algorithm can lead to a
# predictor that achieves a much better accuracy for women than for men, and
# that it is insufficient to simply remove the sensitive feature (in this case
# sex) from training. We then use Fairlearn to mitigate this disparity in
# accuracy with either `ThresholdOptimizer` or `GridSearch`.

# General imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# set layout parameters to avoid cutting off legend
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Data processing
from sklearn.model_selection import train_test_split

# Models
import lightgbm as lgb

# Fairlearn algorithms and utils
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import GridSearch, EqualizedOdds
from fairlearn.widget import FairlearnDashboard

# Metrics
from fairlearn.metrics import (
    selection_rate, demographic_parity_difference, demographic_parity_ratio,
    balanced_accuracy_score_group_summary, roc_auc_score_group_summary,
    equalized_odds_difference, difference_from_summary)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# %%
# The UCI credit card default dataset
# -----------------------------------
#
# The UCI dataset contains data on 30,000 clients and their credit card
# transactions at a bank in Taiwan. In addition to static client features,
# the dataset contains the history of credit card bill payments between April
# and September 2005, as well as the balance limit of the client's credit
# card. The target is whether the client will default on a card payment in the
# following month, October 2005. One can imagine that a model trained on this
# data can be used in practice to determine whether a client is eligible for
# another product such as an auto loan.

# Load the data
data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
dataset = pd.read_excel(io=data_url, header=1).drop(columns=['ID']).rename(columns={'PAY_0': 'PAY_1'})
dataset.head()

# %%
# Dataset columns:
#
# * `LIMIT_BAL`: credit card limit, will be replaced by a synthetic feature
# * `SEX, EDUCATION, MARRIAGE, AGE`: client demographic features
# * `BILL_AMT[1-6]`: amount on bill statement for April-September
# * `PAY_AMT[1-6]`: payment amount for April-September
# * `default payment next month`: target, whether the customer defaulted the
#   following month

# Extract the sensitive feature
A = dataset["SEX"]
A_str = A.map({2: "female", 1: "male"})

# %%
# Extract the target

Y = dataset["default payment next month"]
categorical_features = ['EDUCATION', 'MARRIAGE', 'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for col in categorical_features:
    dataset[col] = dataset[col].astype('category')

# %%
# Introduce a synthetic feature
# *****************************
#
# We manipulate the balance-limit feature `LIMIT_BAL` to make it highly
# predictive for women but not for men. For example, we can imagine that a
# lower credit limit indicates that a female client is less likely to default,
# but provides no information on a male client's probability of default.

dist_scale = 0.5
np.random.seed(12345)
# Make 'LIMIT_BAL' informative of the target
dataset['LIMIT_BAL'] = Y + np.random.normal(scale=dist_scale, size=dataset.shape[0])
# But then make it uninformative for the male clients
dataset.loc[A == 1, 'LIMIT_BAL'] = np.random.normal(scale=dist_scale, size=dataset[A == 1].shape[0])

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
# Plot distribution of LIMIT_BAL for men
dataset['LIMIT_BAL'][(A == 1) & (Y == 0)].plot(kind='kde', label="Payment on Time", ax=ax1,
                                               title="LIMIT_BAL distribution for men")
dataset['LIMIT_BAL'][(A == 1) & (Y == 1)].plot(kind='kde', label="Payment Default", ax=ax1)
# Plot distribution of LIMIT_BAL for women
dataset['LIMIT_BAL'][(A == 2) & (Y == 0)].plot(kind='kde', label="Payment on Time", ax=ax2,
                                               legend=True, title="LIMIT_BAL distribution for women")
dataset['LIMIT_BAL'][(A == 2) & (Y == 1)].plot(kind='kde', label="Payment Default", ax=ax2,
                                               legend=True).legend(bbox_to_anchor=(1.6, 1))
plt.show()

# %%
# We notice from the above figures that the new `LIMIT_BAL` feature is indeed
# highly predictive for women, but not for men.

# Train-test split
df_train, df_test, Y_train, Y_test, A_train, A_test, A_str_train, A_str_test = train_test_split(
    dataset.drop(columns=['SEX', 'default payment next month']),
    Y,
    A,
    A_str,
    test_size=0.3,
    random_state=12345,
    stratify=Y)

# %%
# Using a fairness unaware model
# ------------------------------
#
# We train an out-of-the-box `lightgbm` model on the modified data and assess
# several disparity metrics.

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 10,
    'max_depth': 3
}

model = lgb.LGBMClassifier(**lgb_params)

model.fit(df_train, Y_train)

# Scores on test set
test_scores = model.predict_proba(df_test)[:, 1]

# Train AUC
roc_auc_score(Y_train, model.predict_proba(df_train)[:, 1])

# %%
# Predictions (0 or 1) on test set
test_preds = (test_scores >= np.mean(Y_train)) * 1

# %%
# LightGBM feature importance
lgb.plot_importance(model, height=0.6, title="Features importance (LightGBM)", importance_type="gain", max_num_features=15)
plt.show()

# %%
# We notice that the synthetic feature `LIMIT_BAL` appears as the most
# important feature in this model although it has no predictive power for an
# entire demographic segment in the data.


# Helper functions
def get_metrics_df(models_dict, y_true, group):
    metrics_dict = {
        "Overall selection rate": (
            lambda x: selection_rate(y_true, x), True),
        "Demographic parity difference": (
            lambda x: demographic_parity_difference(y_true, x, sensitive_features=group), True),
        "Demographic parity ratio": (
            lambda x: demographic_parity_ratio(y_true, x, sensitive_features=group), True),
        "-----": (lambda x: "", True),
        "Overall balanced error rate": (
            lambda x: 1-balanced_accuracy_score(y_true, x), True),
        "Balanced error rate difference": (
            lambda x: difference_from_summary(
                balanced_accuracy_score_group_summary(y_true, x, sensitive_features=group)), True),
        "Equalized odds difference": (
            lambda x: equalized_odds_difference(y_true, x, sensitive_features=group), True),
        "------": (lambda x: "", True),
        "Overall AUC": (
            lambda x: roc_auc_score(y_true, x), False),
        "AUC difference": (
            lambda x: difference_from_summary(
                roc_auc_score_group_summary(y_true, x, sensitive_features=group)), False),
    }
    df_dict = {}
    for metric_name, (metric_func, use_preds) in metrics_dict.items():
        df_dict[metric_name] = [metric_func(preds) if use_preds else metric_func(scores)
                                for model_name, (preds, scores) in models_dict.items()]
    return pd.DataFrame.from_dict(df_dict, orient="index", columns=models_dict.keys())


# %%
# We calculate several performance and disparity metrics below:

models_dict = {"Unmitigated": (test_preds, test_scores)}
get_metrics_df(models_dict, Y_test, A_str_test)

# %%
# As the overall performance metric we use the *area under ROC curve* (AUC),
# which is suited to classification problems with a large imbalance between
# positive and negative examples. For binary classifiers, this is the same as
# *balanced accuracy*.
#
# As the fairness metric we use *equalized odds difference*, which quantifies
# the disparity in accuracy experienced by different demographics. Our goal is
# to assure that neither of the two groups (men vs women) has substantially
# larger false-positive rates or false-negative rates than the other group.
# The equalized odds difference is equal to the larger of the following two
# numbers: (1) the difference between false-positive rates of the two groups,
# (2) the difference between false-negative rates of the two groups.
#
# The table above shows the overall AUC of 0.85 (based on continuous
# predictions) and the overall balanced error rate of 0.22 (based on 0/1
# predictions). Both of these are satisfactory in our application context.
# However, there is a large disparity in accuracy rates (as indicated by the
# balanced error rate difference) and even larger when we consider the
# equalized-odds difference. As a sanity check, we also show the demographic
# parity ratio, whose level (slightly above 0.8) is considered satisfactory
# in this context.
#
# Mitigating equalized odds difference with Postprocessing
# --------------------------------------------------------
#
# We attempt to mitigate the disparities in the `lightgbm` predictions using
# the Fairlearn postprocessing algorithm `ThresholdOptimizer`. This algorithm
# finds a suitable threshold for the scores (class probabilities) produced by
# the `lightgbm` model by optimizing the accuracy rate under the constraint
# that the equalized odds difference (on training data) is zero. Since our
# goal is to optimize balanced accuracy, we resample the training data to have
# the same number of positive and negative examples. This means that
# `ThresholdOptimizer` is effectively optimizing balanced accuracy on the
# original data.

postprocess_est = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds")

# %%
# Balanced data set is obtained by sampling the same number of points from
# the majority class (Y=0) as there are points in the minority class (Y=1)
balanced_idx1 = df_train[Y_train == 1].index
pp_train_idx = balanced_idx1.union(Y_train[Y_train == 0].sample(n=balanced_idx1.size, random_state=1234).index)

df_train_balanced = df_train.loc[pp_train_idx, :]
Y_train_balanced = Y_train.loc[pp_train_idx]
A_train_balanced = A_train.loc[pp_train_idx]

postprocess_est.fit(df_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)

postprocess_preds = postprocess_est.predict(df_test, sensitive_features=A_test)

models_dict = {"Unmitigated": (test_preds, test_scores),
               "ThresholdOptimizer": (postprocess_preds, postprocess_preds)}
get_metrics_df(models_dict, Y_test, A_str_test)

# %%
# The `ThresholdOptimizer` algorithm significantly reduces the disparity
# according to multiple metrics. However, the performance metrics (balanced
# error rate as well as AUC) get worse. Before deploying such a model in
# practice, it would be important to examine in more detail why we observe
# such a sharp trade-off. In our case it is because the available features are
# much less informative for one of the demographic groups than for the other.
#
# Note that unlike the unmitigated model, `ThresholdOptimizer` produces 0/1
# predictions, so its balanced error rate difference is equal to the AUC
# difference, and its overall balanced error rate is equal to 1 - overall AUC.
#
# Below, we compare this model with the unmitigated `lightgbm` model using
# the Fairlearn dashboard. As the performance metric, we can select the
# balanced accuracy. The dashboard right now does not directly show the
# equalized odds difference, but a similar information is shown in the
# *Disparity in Accuracy* view, where we can examine the difference between
# overprediction and underprediction rates of the two groups.

# **Unmitigated Model vs ThresholdOptimizer: Dashboard Demo**

FairlearnDashboard(sensitive_features=A_str_test,
                   sensitive_feature_names=['Sex'],
                   y_true=Y_test,
                   y_pred={"Unmitigated": test_preds,
                           "ThresholdOptimizer": postprocess_preds})

# %%
# Mitigating equalized odds difference with GridSearch
# ----------------------------------------------------
# We now attempt to mitigate disparities using the `GridSearch` algorithm.
# Unlike `ThresholdOptimizer`, the predictors produced by `GridSearch` do not
# access the sensitive feature at test time. Also, rather than training a
# single model, we train multiple models corresponding to different trade-off
# points between the performance metric (balanced accuracy) and fairness
# metric (equalized odds difference).

# Train GridSearch
sweep = GridSearch(model,
                   constraints=EqualizedOdds(),
                   grid_size=41,
                   grid_limit=2)

sweep.fit(df_train_balanced, Y_train_balanced, sensitive_features=A_train_balanced)

sweep_preds = [predictor.predict(df_test) for predictor in sweep.predictors_]
sweep_scores = [predictor.predict_proba(df_test)[:, 1] for predictor in sweep.predictors_]

equalized_odds_sweep = [
    equalized_odds_difference(Y_test, preds, sensitive_features=A_str_test)
    for preds in sweep_preds
]
balanced_accuracy_sweep = [balanced_accuracy_score(Y_test, preds) for preds in sweep_preds]
auc_sweep = [roc_auc_score(Y_test, scores) for scores in sweep_scores]

# %%
# Select only non-dominated models (with respect to balanced accuracy and
# equalized odds difference)
all_results = pd.DataFrame(
    {"predictor": sweep.predictors_, "accuracy": balanced_accuracy_sweep, "disparity": equalized_odds_sweep}
)
non_dominated = []
for row in all_results.itertuples():
    accuracy_for_lower_or_eq_disparity = all_results["accuracy"][all_results["disparity"] <= row.disparity]
    if row.accuracy >= accuracy_for_lower_or_eq_disparity.max():
        non_dominated.append(True)
    else:
        non_dominated.append(False)

equalized_odds_sweep_non_dominated = np.asarray(equalized_odds_sweep)[non_dominated]
balanced_accuracy_non_dominated = np.asarray(balanced_accuracy_sweep)[non_dominated]
auc_non_dominated = np.asarray(auc_sweep)[non_dominated]

# %%
# Plot equalized odds difference vs balanced accuracy
plt.scatter(balanced_accuracy_non_dominated,
            equalized_odds_sweep_non_dominated,
            label="GridSearch Models")
plt.scatter(balanced_accuracy_score(Y_test, test_preds),
            equalized_odds_difference(Y_test, test_preds, sensitive_features=A_str_test),
            label="Unmitigated Model")
plt.scatter(balanced_accuracy_score(Y_test, postprocess_preds),
            equalized_odds_difference(Y_test, postprocess_preds, sensitive_features=A_str_test),
            label="ThresholdOptimizer Model")
plt.xlabel("Balanced Accuracy")
plt.ylabel("Equalized Odds Difference")
plt.legend(bbox_to_anchor=(1.55, 1))
plt.show()

# %%
# As intended, `GridSearch` models appear along the trade-off curve between
# the large balanced accuracy (but also large disparity), and low disparity
# (but worse balanced accuracy). This gives the data scientist a flexibility
# to select a model that fits the application context best.

# Plot equalized odds difference vs AUC
plt.scatter(auc_non_dominated,
            equalized_odds_sweep_non_dominated,
            label="GridSearch Models")
plt.scatter(roc_auc_score(Y_test, test_scores),
            equalized_odds_difference(Y_test, test_preds, sensitive_features=A_str_test),
            label="Unmitigated Model")
plt.scatter(roc_auc_score(Y_test, postprocess_preds),
            equalized_odds_difference(Y_test, postprocess_preds, sensitive_features=A_str_test),
            label="ThresholdOptimizer Model")
plt.xlabel("AUC")
plt.ylabel("Equalized Odds Difference")
plt.legend(bbox_to_anchor=(1.55, 1))
plt.show()

# %%
# Similarly, `GridSearch` models appear along the trade-off curve between AUC
# and equalized odds difference.

model_sweep_dict = {"GridSearch_{}".format(i): sweep_preds[i] for i in range(len(sweep_preds)) if non_dominated[i]}
model_sweep_dict.update({
    "Unmitigated": test_preds,
    "ThresholdOptimizer": postprocess_preds
})

# %%
# **Grid Search: Dashboard Demo**
# We compare the `GridSearch` candidate models with the unmitigated `lightgbm`
# model and the `ThresholdOptimizer` model using the Fairlearn dashboard. We
# can select the balanced accuracy as the performance metric and examine the
# trade-off curve between balanced accuracy and the disparity in balanced
# accuracy.

FairlearnDashboard(sensitive_features=A_str_test,
                   sensitive_feature_names=['Sex'],
                   y_true=Y_test,
                   y_pred=model_sweep_dict)
