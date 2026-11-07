# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
==============================
Plotting with control features
==============================
"""

# %%
# A :class:`~fairlearn.metrics.MetricFrame` accepts ``control_features`` in
# addition to ``sensitive_features``. Both split the data into groups, but
# aggregations such as :meth:`~fairlearn.metrics.MetricFrame.difference` are
# computed *within* each group of the control features, rather than across them.
# This is useful when some differences between groups are expected, for example
# between age groups, and the question is whether the metrics still differ
# between the sensitive groups inside each age group.
#
# This example shows how to plot the metrics of such a
# :class:`~fairlearn.metrics.MetricFrame`. It only illustrates the mechanics: it
# makes no claim about whether the classifier is fair.

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame, count, selection_rate

# %%
# Load the data
# -------------
# We use the UCI Adult dataset, where the task is to predict whether a person
# earns more than $50,000 a year. The sensitive feature is ``sex``. The
# control feature is ``age``, which we group into three bands. The groups of a
# :class:`~fairlearn.metrics.MetricFrame` are listed in the sorted order of
# their labels, so we choose labels that sort in the order of the ages.

X, y = fetch_adult(return_X_y=True)
y = y == ">50K"
sex = X["sex"]
age_group = pd.cut(X["age"], bins=[0, 30, 50, 100], right=False, labels=["17-29", "30-49", "50+"])

X_train, X_test, y_train, y_test, sex_train, sex_test, age_train, age_test = train_test_split(
    pd.get_dummies(X), y, sex, age_group, test_size=0.3, random_state=123
)

classifier = DecisionTreeClassifier(min_samples_leaf=20, max_depth=5, random_state=123)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# %%
# Compute the metrics
# -------------------
# The age group is passed as ``control_features``. We include
# :func:`~fairlearn.metrics.count` as a metric, because the groups within an
# age band can be small, and small groups make the other metrics noisy.

# Compute metrics with control features
metric_frame = MetricFrame(
    metrics={"selection rate": selection_rate, "accuracy": accuracy_score, "count": count},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sex_test,
    control_features=age_test,
)
# End compute metrics with control features

# %%
# The ``overall`` property now has one row per age group, instead of a single
# row:
metric_frame.overall

# %%
# The ``by_group`` property has a two-level index, with the control feature at
# the top level:
metric_frame.by_group

# %%
# Plot all groups
# ---------------
# Since ``by_group`` is a :class:`pandas.DataFrame`, it can be plotted in the
# same way as without control features. However, every combination of age group
# and sex is now drawn as an independent bar, and nothing in the plot shows
# which bars belong to the same age group.

# Plot all groups side by side
metric_frame.by_group[["selection rate", "accuracy"]].plot.bar(
    subplots=True,
    layout=[1, 2],
    legend=False,
    figsize=[12, 5],
    title="Metrics for each combination of age group and sex",
)
plt.tight_layout()
# End plot all groups side by side

# %%
# Plot the groups of each control group together
# ----------------------------------------------
# To compare the sensitive groups *within* each control group, move the
# sensitive feature from the index to the columns with
# :meth:`pandas.DataFrame.unstack`. The bars are then clustered by age group,
# with one bar per sex in each cluster.

# Cluster the bars by control group
metric_frame.by_group["selection rate"].unstack(level="sex").plot.bar(
    rot=0,
    ylabel="selection rate",
    figsize=[8, 5],
    title="Selection rate by sex within each age group",
)
# End cluster the bars by control group

# %%
# Compare the groups within each control group
# --------------------------------------------
# The same grouping applies to the aggregations. Instead of a single number,
# :meth:`~fairlearn.metrics.MetricFrame.difference` returns the largest
# difference between the sexes for each age group, which is the value to read
# next to the clustered bars above:
metric_frame.difference()[["selection rate", "accuracy"]]
