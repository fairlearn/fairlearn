# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
==============================
Metrics with Multiple Features
==============================
"""
# %%
# This notebook demonstrates the new API for metrics, which supports
# multiple sensitive and conditional features.
#
# Getting the Data
# ================
#
# To demonstrate the API, we use the well-known 'Adult' dataset,
# and we train a simple model on it. We start with some
# uncontroversial `import` statements:

import functools
import sklearn.metrics as skm
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.metrics import GroupedMetric, make_derived_metric
from fairlearn.metrics import accuracy_score_group_min, accuracy_score_difference


# %%
# Next, we import the data, dropping some of the values to
# help maintain clarity:

data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
X_raw = X_raw[X_raw.race != 'Other']
Y = data.target[data.data.race != 'Other']
Y = (Y == '>50K') * 1

# %%
# We can select some columns which we might want to use for
# our sensitive and conditional features:


def marriage_transform(m_s_string):
    """Perform some simple manipulations."""
    result = 'A'
    if m_s_string.startswith("Married"):
        result = 'B'
    elif m_s_string.startswith("Widowed"):
        result = 'C'
    return result


def occupation_transform(occ_string):
    """Perform some simple manipulations."""
    result = 'pp'
    if occ_string.startswith("Machine"):
        result = 'qq'
    return result


colA = X_raw['marital-status'].map(marriage_transform).fillna('C')
colA.name = "Feature A"
colB = X_raw['occupation'].map(occupation_transform).fillna('qq')
colB.name = "Feature B"

A = X_raw[['race', 'sex']]
A['Feature A'] = colA
A['Feature B'] = colB
A

# %%
# With the data imported, we perform some standard processing, and a test/train split:
le = LabelEncoder()
Y = le.fit_transform(Y)

le = LabelEncoder()

sc = StandardScaler()
X_dummies = pd.get_dummies(X_raw)
X_scaled = sc.fit_transform(X_dummies)
X_scaled = pd.DataFrame(X_scaled, columns=X_dummies.columns)

X_train, X_test, Y_train, Y_test, A_train, A_test = \
    train_test_split(X_scaled, Y, A,
                     test_size=0.3,
                     random_state=12345,
                     stratify=Y)

# Ensure indices are aligned
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# %%
# Finally, we train a simple model on the data, and generate
# some predictions:


unmitigated_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)
unmitigated_predictor.fit(X_train, Y_train)

Y_pred = unmitigated_predictor.predict(X_test)

# %%
# Simple Metrics
# ==============
#
# We now start computing metrics. The new API is based around the
# `GroupedMetric` object. The constructor specifies the data, the
# metric and the sensitive feature:


basic_metric = GroupedMetric(skm.recall_score, Y_test, Y_pred, sensitive_features=A_test['sex'])

# %%
# The `GroupedMetric` object has properties of `overall` and
# `by_group`, which show the overall value of the metric (evaluated
# on the entire dataset), as well as the metric evaluated on each
# of the unique values of the specified sensitive feature:

basic_metric.overall

# %%
# This can be compared to a direct calculation from SciKit-Learn:
skm.recall_score(Y_test, Y_pred)

# %%
# The results for each value of the sensitive feature:
basic_metric.by_group

# %%
# Suppose that we have sample weights which we want to use in the
# metric calculation. For this we use the `sample_params=` argument
# in the constructor. This contains a dictionary of arrays which
# need to be sliced up with the `y_true` and `y_pred` arrays, before
# being passed into the metric function. The dictionary keys are the
# names of the arguments in the metric function signature:

wgts = np.random.random(size=len(Y_test))

basic_metric_wgts = GroupedMetric(skm.recall_score,
                                  Y_test, Y_pred,
                                  sensitive_features=A_test['sex'],
                                  sample_params={'sample_weight': wgts})

# %%
# The overall values
basic_metric_wgts.overall

# %%
# And compare with SciKit-Learn:
skm.recall_score(Y_test, Y_pred, sample_weight=wgts)

# %%
# The values for the sensitive feature:
basic_metric_wgts.by_group

# %%
# If the metric function has other arguments, then it will need to be
# wrapped. An example is `fbeta_score()` which requires a value for
# `beta`. The `functools.partial` routine makes this easy:
fbeta_05 = functools.partial(skm.fbeta_score, beta=0.5)

basic_metric_wrapped = GroupedMetric(fbeta_05, Y_test, Y_pred, sensitive_features=A_test['sex'])

# %%
# The overall values
basic_metric_wrapped.overall

# %%
# The Values for the sensitive feature
basic_metric_wrapped.by_group

# %%
# We can evaluate multiple metrics at once by passing in a dictionary
# of metric functions. If we have sample parameters as well, then
# that argument becomes a dictionary of dictionaries, with the top
# set of keys matching those in the metrics dictionary:

metric_dict = {'recall': skm.recall_score, 'fbeta_0.5': fbeta_05}
sample_params = {'recall': {'sample_weight': wgts}, 'fbeta_0.5': {'sample_weight': wgts}}

basic_metric_two = GroupedMetric(metric_dict,
                                 Y_test, Y_pred,
                                 sensitive_features=A_test['sex'],
                                 sample_params=sample_params)

# %%
# Now we inspect the overall values
basic_metric_two.overall

# %%
# And the values for the sensitive feature
basic_metric_two.by_group

# %%
# Aggregates
# ----------
#
# We provide some aggregating functions, which provide means of
# obtaining scalar measures. First are the `group_min()` and
# `group_max()` methods which compute the minimum and maximum
# values of each metric across the sensitive feature.
#
# First the minimum:
basic_metric_two.group_min()

# %%
# And the maximum:
basic_metric_two.group_max()

# %%
# There is also a `difference()` method, which calculates the
# difference based on its `method=` argument.
#
# First the default difference between the minimum and maximum:
basic_metric_two.difference(method='minmax')

# %%
# And the difference to the overall value:
basic_metric_two.difference(method='to_overall')

# %%
# Multiple Sensitive Features
# ===========================
#
# The new metrics are not constrained to considering a
# single sensitive feature at a time. Multiple columns can
# be passed into as senstive features, and the intersections
# of all subgroups will be computed:

two_sf = GroupedMetric(metric_dict,
                       Y_test, Y_pred,
                       sensitive_features=A_test[['sex', 'race']])

# %%
# The overall values
two_sf.overall

# %%
# The values for the intersections of the sensitive features:
two_sf.by_group

# %%
# The maximum absolute difference to the overall value for each
# intersection of the sensitive features:
two_sf.difference(method='to_overall')

# %%
# Conditional Features
# ====================
#
# Conditional features denote groups for which outcomes are
# allowed to differ. For example, in a loan scenario, it is
# acceptable for people in a high income group to be offered
# loans more often than those in a low income group. While
# this could be monitored by splitting the input array,
# `GroupedMetric` supports this directly:

cond_metric = GroupedMetric(skm.recall_score,
                            Y_test, Y_pred,
                            sensitive_features=A_test[['sex', 'race']],
                            conditional_features=A_test['Feature A'])

# %%
# This changes the `overall` property to be a DataFrame. The
# rows correspond to the unique values of the conditional
# feature:

cond_metric.overall

# %%
# The `by_group` property still looks similar - indeed,
# can compare it to a metric which moves the conditional
# feature into the sensitive feature list.
#
# First computed from the object above:
cond_metric.by_group

# %%
# Now recompute, with the conditional feature moved to the
# list of sensitive features:
cond_metric_alt = GroupedMetric(skm.recall_score,
                                Y_test, Y_pred,
                                sensitive_features=A_test[['Feature A', 'sex', 'race']])
cond_metric_alt.by_group

# %%
# The aggregates are also evaluated for each unique value
# of the conditional feature. For example the maximum:
cond_metric.group_max()

# %%
# And the difference:
cond_metric.difference(method='minmax')

# %%
# We also support multiple conditional features, and
# evaluate multiple metrics at once:

cond_metric_two = GroupedMetric(metric_dict,
                                Y_test, Y_pred,
                                sensitive_features=A_test[['sex', 'race']],
                                conditional_features=A_test[['Feature A', 'Feature B']])

# %%
# First the overall values
cond_metric_two.overall

# %%
# The values for the intersections of sensitive and conditional features:
cond_metric_two.by_group

# %%
# The maximum absolute values of the difference to the overall value
# for each intersection of conditional features:
cond_metric_two.difference(method='to_overall')

# %%
# Obtaining Scalars
# =================
#
# It is quite common for a training pipeline to require scalar
# metrics in order to monitor progress. To this end, we provide
# a 'make_derived_metric()` routine, which creates a scalar-production
# metric function based on a given underlying metric and
# an aggregation function:
calc_min_accuracy = make_derived_metric('group_min',
                                        skm.accuracy_score,
                                        sample_param_names=['sample_weight'])

# %%
# This then acts like any other metric:
calc_min_accuracy(Y_test, Y_pred, sensitive_features=A_test['sex'])
# %%
# Internally, this has called the `.difference()` method and extracted the
# single result. Compare:
acc_group = GroupedMetric(skm.accuracy_score,
                          Y_test, Y_pred,
                          sensitive_features=A_test['sex'])
acc_group.group_min()
# %%
# Aggregations of `'group_min'`, `'group_max'`, `'difference'`, and
# `'ratio'` are supported. For the last two, the resulting function
# requires a `method=` argument to specify how to calculate the final
# value:
calc_acc_diff = make_derived_metric('difference',
                                    skm.accuracy_score,
                                    sample_param_names=['sample_weight'])
acc_from_func = calc_acc_diff(Y_test, Y_pred,
                              sensitive_features=A_test['sex'],
                              method='minmax')
acc_from_gm = acc_group.difference(method='minmax')['accuracy_score']
print("Calculation from derived function: ", acc_from_func)
print("Calculation from GroupedMetric:", acc_from_gm)
assert acc_from_func == acc_from_gm
# %%
# Conditional features are obviously not supported, but sample parameters
# (such as weights) are:
calc_acc_diff(Y_test, Y_pred,
              method='minmax',
              sensitive_features=A_test['sex'],
              sample_weight=wgts)

# %%
# As a convenience, we have pre-wrapped a number of metrics from SciKit-Learn.
# For example:
pre_wrapped_acc_min = accuracy_score_group_min(Y_test, Y_pred,
                                               sensitive_features=A_test['sex'])
pre_wrapped_acc_diff = accuracy_score_difference(Y_test, Y_pred,
                                                 sensitive_features=A_test['sex'],
                                                 method='minmax')
print("Prewrapped Min: ", pre_wrapped_acc_min)
print("Prewrapped Diff: ", pre_wrapped_acc_diff)
assert pre_wrapped_acc_diff == acc_from_func
