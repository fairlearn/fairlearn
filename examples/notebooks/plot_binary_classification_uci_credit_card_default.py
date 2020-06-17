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
