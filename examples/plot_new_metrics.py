# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
==============================
Metrics with Multiple Features
==============================
"""
# %%
# This notebook demonstrates the new API for metrics, which supports
# multiple sensitive and conditional features. This example does not
# contain a proper discussion of how fairness relates to the dataset
# used, although it does highlight issues which users may want to
# consider when analysing their datasets.
#
# We are going to consider a lending scenario, supposing that we have
# a model which predicts whether or not a particular customer will
# repay a loan. This could be used as the basis of deciding whether
# or not to offer that customer a loan. With traditional metrics,
# we would assess the model using:
#
# - The 'true' values from the test set
# - The model predictions from the test set
#
# Our fairness metrics compute group-based fairness statistics.
# To use these, we also need categorical columns from the test
# set. For this example, we will include:
#
# - The sex of each individual (two unique values)
# - The race of each individual (three unique values)
# - The credit score band of each individual (three unique values)
# - Whether the loan is considered 'large' or 'small'
#
# An individual's sex and race should not affect a lending decision,
# but it would be legitimate to consider an individual's credit score
# and the relative size of the loan which they desired.
#
# A real scenario will be more complicated, but this will serve to
# illustrate the use of the new metrics.
#
# Getting the Data
# ================
#
# *This section may be skipped. It simply creates a dataset for
# illustrative purposes*
#
# We will use the well-known UCI 'Adult' dataset as the basis of this
# demonstration. This is not for a lending scenario, but we will regard
# it as one for the purposes of this example. We will use the existing
# 'race' and 'sex' columns (trimming the former to three unique values),
# and manufacture credit score bands and loan sizes from other columns.
# We start with some uncontroversial `import` statements:

import functools
import numpy as np

import sklearn.metrics as skm
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline

from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate, count


# %%
# Next, we import the data:

data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
y = (data.target == '>50K') * 1

# %%
# For purposes of clarity, we consolidate the 'race' column to have
# three unique values:


def race_transform(input_str):
    """Reduce values to White, Black and Other."""
    result = 'Other'
    if input_str == 'White' or input_str == 'Black':
        result = input_str
    return result


X_raw['race'] = X_raw['race'].map(race_transform).fillna('Other').astype('category')
print(np.unique(X_raw['race']))

# %%
# Now, we manufacture the columns for the credit score band and
# requested loan size. These are wholly constructed, and not
# part of the actual dataset in any way. They are simply for
# illustrative purposes.


def marriage_transform(m_s_string):
    """Perform some simple manipulations."""
    result = 'Low'
    if m_s_string.startswith("Married"):
        result = 'Medium'
    elif m_s_string.startswith("Widowed"):
        result = 'High'
    return result


def occupation_transform(occ_string):
    """Perform some simple manipulations."""
    result = 'Small'
    if occ_string.startswith("Machine"):
        result = 'Large'
    return result


col_credit = X_raw['marital-status'].map(marriage_transform).fillna('Low')
col_credit.name = "Credit Score"
col_loan_size = X_raw['occupation'].map(occupation_transform).fillna('Small')
col_loan_size.name = "Loan Size"

A = X_raw[['race', 'sex']]
A['Credit Score'] = col_credit
A['Loan Size'] = col_loan_size
A

# %%
# Now that we have imported our dataset and manufactured a few features, we
# can perform some more conventional processing. To avoid the problem of
# `data leakage <https://en.wikipedia.org/wiki/Leakage_(machine_learning)>`_,
# we need to split the data into training and test sets before applying
# any transforms or scaling:


(X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
    X_raw, y, A, test_size=0.3, random_state=54321, stratify=y
)

# Ensure indices are aligned between X, y and A,
# after all the slicing and splitting of DataFrames
# and Series

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# %%
# Next, we build two :class:`~sklearn.pipeline.Pipeline` objects
# to process the columns, one for numeric data, and the other
# for categorical data. Both impute missing values; the difference
# is whether the data are scaled (numeric columns) or
# one-hot encoded (categorical columns). Imputation of missing
# values should generally be done with care, since it could
# potentially introduce biases. Of course, removing rows with
# missing data could also cause trouble, if particular subgroups
# have poorer data quality.

numeric_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
)
categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)

# %%
# With our preprocessor defined, we can now build a
# new pipeline which includes an Estimator:

unmitigated_predictor = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(solver="liblinear", fit_intercept=True),
        ),
    ]
)

# %%
# With the pipeline fully defined, we can first train it
# with the training data, and then generate predictions
# from the test data.

unmitigated_predictor.fit(X_train, y_train)
y_pred = unmitigated_predictor.predict(X_test)


# %%
# Analysing the Model with Metrics
# ================================
#
# After our data manipulations and model training, we have the following
# from our test set:
#
# - A vector of true values called ``y_test``
# - A vector of model predictions called ``y_pred``
# - A DataFrame of categorical features relevant to fairness called ``A_test``
#
# In a traditional model analysis, we would now look at some metrics
# evaluated on the entire dataset. Suppose in this case, the relevant
# metrics are :func:`fairlearn.metrics.selection_rate` and
# :func:`sklearn.metrics.fbeta_score` (with
# ``beta=0.6``).
# We can evaluate these metrics directly:

print("Selection Rate:", selection_rate(y_test, y_pred))
print("fbeta:", skm.fbeta_score(y_test, y_pred, beta=0.6))

# %%
# We know that there are sensitive features in our data, and we want to
# ensure that we're not harming individuals due to membership in any of
# these groups. For this purpose, Fairlearn provides the
# :class:`fairlearn.metrics.MetricFrame`
# class. Let us construct an instance of this class, and then look at
# its capabilities:

fbeta_06 = functools.partial(skm.fbeta_score, beta=0.6)

metric_fns = {'selection_rate': selection_rate, 'fbeta_06': fbeta_06, 'count': count}

grouped_on_sex = MetricFrame(metrics=metric_fns,
                             y_true=y_test,
                             y_pred=y_pred,
                             sensitive_features=A_test['sex'])

# %%
# The :class:`fairlearn.metrics.MetricFrame` object requires a
# minimum of four arguments:
#
# 1. The underlying metric function(s) to be evaluated
# 2. The true values
# 3. The predicted values
# 4. The sensitive feature values
#
# These are all passed as arguments to the constructor. If more than
# one underlying metric is required (as in this case), then we must
# provide them in a dictionary.
#
# The underlying metrics must have a signature ``fn(y_true, y_pred)``,
# so we have to use :func:`functools.partial` on ``fbeta_score()`` to
# furnish ``beta=0.6`` (we will show how to pass in extra array
# arguments such as sample weights shortly).
#
# We will now take a closer look at the :class:`fairlearn.metrics.MetricFrame`
# object. First, there is the ``overall`` property, which contains
# the metrics evaluated on the entire dataset. We see that this contains the
# same values calculated above:

assert grouped_on_sex.overall['selection_rate'] == selection_rate(y_test, y_pred)
assert grouped_on_sex.overall['fbeta_06'] == skm.fbeta_score(y_test, y_pred, beta=0.6)
print(grouped_on_sex.overall)

# %%
# The other property in the :class:`fairlearn.metrics.MetricFrame` object
# is ``by_group``. This contains the metrics evaluated on each subgroup defined
# by the categories in the ``sensitive_features=`` argument. Note that
# :func:`fairlearn.metrics.count` can be used to display the number of
# data points in each subgroup. In this case, we have results for males and females:

grouped_on_sex.by_group

# %%
# We can immediately see a substantial disparity in the selection rate between
# males and females.
#
# We can also create another :class:`fairlearn.metrics.MetricFrame` object
# using race as the sensitive feature:

grouped_on_race = MetricFrame(metrics=metric_fns,
                              y_true=y_test,
                              y_pred=y_pred,
                              sensitive_features=A_test['race'])

# %%
# The ``overall`` property is unchanged:
assert (grouped_on_sex.overall == grouped_on_race.overall).all()

# %%
# The ``by_group`` property now contains the metrics evaluated based on the 'race'
# column:
grouped_on_race.by_group

# %%
# We see that there is also a significant disparity in selection rates when
# grouping by race.

# %%
# Sample weights and other arrays
# -------------------------------
#
# We noted above that the underlying metric functions passed to the
# :class:`fairlearn.metrics.MetricFrame` constructor need to be of
# the form ``fn(y_true, y_pred)`` - we do not support scalar arguments
# such as ``pos_label=`` or ``beta=`` in the constructor. Such
# arguments should be bound into a new function using
# :func:`functools.partial`, and the result passed in. However, we do
# support arguments which have one entry for each sample, with an array
# of sample weights being the most common example. These are divided
# into subgroups along with ``y_true`` and ``y_pred``, and passed along
# to the underlying metric.
#
# To use these arguments, we pass in a dictionary as the ``sample_params=``
# argument of the constructor. Let us generate some random weights, and
# pass these along:

random_weights = np.random.rand(len(y_test))

example_sample_params = {
    'selection_rate': {'sample_weight': random_weights},
    'fbeta_06': {'sample_weight': random_weights},
}


grouped_with_weights = MetricFrame(metrics=metric_fns,
                                   y_true=y_test,
                                   y_pred=y_pred,
                                   sensitive_features=A_test['sex'],
                                   sample_params=example_sample_params)

# %%
# We can inspect the overall values, and check they are as expected:
assert grouped_with_weights.overall['selection_rate'] == \
    selection_rate(y_test, y_pred, sample_weight=random_weights)
assert grouped_with_weights.overall['fbeta_06'] == \
    skm.fbeta_score(y_test, y_pred, beta=0.6, sample_weight=random_weights)
print(grouped_with_weights.overall)

# %%
# We can also see the effect on the metric being evaluated on the subgroups:
grouped_with_weights.by_group

# %%
# Quantifying Disparities
# =======================
#
# We now know that our model is selecting individuals who are female far less
# often than individuals who are male. There is a similar effect when
# examining the results by race, with blacks being selected far less often than
# whites (and those classified as 'other'). However, there are many cases where
# presenting all these numbers at once will not be useful (for example, a high
# level dashboard which is monitoring model performance). Fairlearn provides
# several means of aggregating metrics across the subgroups, so that disparities
# can be readily quantified.
#
# The simplest of these aggregations is ``group_min()``, which reports the
# minimum value seen for a subgroup for each underlying metric (we also provide
# ``group_max()``). This is
# useful if there is a mandate that "no subgroup should have an ``fbeta_score()``
# of less than 0.6." We can evaluate the minimum values easily:
grouped_on_race.group_min()

# %%
# As noted above, the selection rates varies greatly by race and by sex.
# This can be quantified in terms of a difference between the subgroup with
# the highest value of the metric, and the subgroup with the lowest value.
# For this, we provide the method ``difference(method='between_groups)``:
grouped_on_race.difference(method='between_groups')

# %%
# We can also evaluate the difference relative to the corresponding overall
# value of the metric. In this case we take the absolute value, so that the
# result is always positive:
grouped_on_race.difference(method='to_overall')

# %%
# There are situations where knowing the ratios of the metrics evaluated on
# the subgroups is more useful. For this we have the ``ratio()`` method.
# We can take the ratios between the minimum and maximum values of each metric:
grouped_on_race.ratio(method='between_groups')

# %%
# We can also compute the ratios relative to the overall value for each
# metric. Analogous to the differences, the ratios are always in the range
# :math:`[0,1]`:
grouped_on_race.ratio(method='to_overall')

# %%
# Intersections of Features
# =========================
#
# So far we have only considered a single sensitive feature at a time,
# and we have already found some serious issues in our example data.
# However, sometimes serious issues can be hiding in intersections of
# features. For example, the
# `Gender Shades project <https://www.media.mit.edu/projects/gender-shades/overview/>`_
# found that facial recognition algorithms performed worse for blacks
# than whites, and also worse for women than men (despite overall high
# accuracy score). Moreover, performance on black females was *terrible*.
# We can examine the intersections of sensitive features by passing
# multiple columns to the :class:`fairlearn.metrics.MetricFrame`
# constructor:

grouped_on_race_and_sex = MetricFrame(metrics=metric_fns,
                                      y_true=y_test,
                                      y_pred=y_pred,
                                      sensitive_features=A_test[['race', 'sex']])

# %%
# The overall values are unchanged, but the ``by_group`` table now
# shows the intersections between subgroups:
assert (grouped_on_race_and_sex.overall == grouped_on_race.overall).all()
grouped_on_race_and_sex.by_group

# %%
# The aggregations are still performed across all subgroups for each metric,
# so each continues to reduce to a single value. If we look at the
# ``group_min()``, we see that we violate the mandate we specified for the
# ``fbeta_score()`` suggested above (for females with a race of 'Other' in
# fact):
grouped_on_race_and_sex.group_min()

# %%
# Looking at the ``ratio()`` method, we see that the disparity is worse
# (specifically between white males and black females, if we check in
# the ``by_group`` table):
grouped_on_race_and_sex.ratio(method='between_groups')

# %%
# Control Features
# ================
#
# There is a further way we can slice up our data. We have (*completely
# made up*) features for the individuals' credit scores (in three bands)
# and also the size of the loan requested (large or small). In our loan
# scenario, it is acceptable that individuals with high credit scores
# are selected more often than individuals with low credit scores.
# However, within each credit score band, we do not want a disparity
# between (say) black females and white males. To example these cases,
# we have the concept of *control features*.
#
# Control features are introduced by the ``control_features=``
# argument to the :class:`fairlearn.metrics.MetricFrame` object:
cond_credit_score = MetricFrame(metrics=metric_fns,
                                y_true=y_test,
                                y_pred=y_pred,
                                sensitive_features=A_test[['race', 'sex']],
                                control_features=A_test['Credit Score'])

# %%
# This has an immediate effect on the ``overall`` property. Instead
# of having one value for each metric, we now have a value for each
# unique value of the control feature:
cond_credit_score.overall

# %%
# The ``by_group`` property is similarly expanded:
cond_credit_score.by_group

# %%
# The aggregates are also evaluated once for each group identified
# by the control feature:
cond_credit_score.group_min()

# %%
# And:
cond_credit_score.ratio(method='between_groups')

# %%
# In our data, we see that we have a dearth of positive results
# for high income non-whites, which significantly affects the
# aggregates.
#
# We can continue adding more control features:
cond_both = MetricFrame(metrics=metric_fns,
                        y_true=y_test,
                        y_pred=y_pred,
                        sensitive_features=A_test[['race', 'sex']],
                        control_features=A_test[['Loan Size', 'Credit Score']])

# %%
# The ``overall`` property now splits into more values:
cond_both.overall

# %%
# As does the ``by_groups`` property, where ``NaN`` values
# indicate that there were no samples in the cell:
cond_both.by_group

# %%
# The aggregates behave similarly. By this point, we are having significant issues
# with under-populated intersections. Consider:


def member_counts(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return len(y_true)


counts = MetricFrame(metrics=member_counts,
                     y_true=y_test,
                     y_pred=y_pred,
                     sensitive_features=A_test[['race', 'sex']],
                     control_features=A_test[['Loan Size', 'Credit Score']])

counts.by_group

# %%
# Recall that ``NaN`` indicates that there were no individuals
# in a cell - ``member_counts()`` will not even have been called.
