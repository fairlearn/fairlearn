# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
======================
Making Derived Metrics
======================
"""
# %%
# This notebook demonstrates the use of the :func:`fairlearn.metrics.make_derived_metric`
# function.
# Many higher-order machine learning algorithms (such as hyperparameter tuners) make use
# of scalar metrics when deciding how to proceed.
# While the :class:`fairlearn.metrics.MetricFrame` has the ability to produce such
# scalars through its aggregation functions, its API does not conform to that usually
# expected by these algorithms.
# The :func:`~fairlearn.metrics.make_derived_metric` function exists to bridge this gap.
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
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairlearn.datasets import fetch_adult
from fairlearn.metrics import MetricFrame, accuracy_score_group_min, make_derived_metric

# %%
# Next, we import the data, dropping any rows which are missing data:

data = fetch_adult()
X_raw = data.data
y = (data.target == ">50K") * 1
A = X_raw[["race", "sex"]]

# %%
# We are now going to preprocess the data. Before applying any transforms,
# we first split the data into train and test sets. All the transforms we
# apply will be trained on the training set, and then applied to the test
# set. This ensures that data doesn't leak between the two sets (this is
# a serious but subtle
# `problem in machine learning <https://en.wikipedia.org/wiki/Leakage_(machine_learning)>`_).
# So, first we split the data:

(X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
    X_raw, y, A, test_size=0.3, random_state=12345, stratify=y
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
# Creating a derived metric
# =========================
#
# Suppose our key metric is the accuracy score, and we are most interested in
# ensuring that it exceeds some threshold for all subgroups
# We might use the :class:`~fairlearn.metrics.MetricFrame` as
# follows:

acc_frame = MetricFrame(
    metrics=skm.accuracy_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
)
print("Minimum accuracy_score: ", acc_frame.group_min())

# %%
# We can create a function to perform this in a single call
# using :func:`~fairlearn.metrics.make_derived_metric`.
# This takes the following arguments (which must always be
# supplied as keyword arguments):
#
# - :code:`metric=`, the base metric function
# - :code:`transform=`, the name of the aggregation
#   transformation to perform. For this demonstration, we
#   want this to be :code:`'group_min'`
# - :code:`sample_param_names=`, a list of parameter names
#   which should be treated as sample
#   parameters. This is optional, and defaults to
#   :code:`['sample_weight']` which is appropriate for many
#   metrics in `scikit-learn`.
#
# The result is a new function with the same signature as the
# base metric, which accepts two extra arguments:
#
#  - :code:`sensitive_features=` to specify the sensitive features
#    which define the subgroups
#  - :code:`method=` to adjust how the aggregation transformation
#    operates. This corresponds to the same argument in
#    :meth:`fairlearn.metrics.MetricFrame.difference` and
#    :meth:`fairlearn.metrics.MetricFrame.ratio`
#
# For the current case, we do not need the :code:`method=`
# argument, since we are taking the minimum value.

my_acc = make_derived_metric(metric=skm.accuracy_score, transform="group_min")
my_acc_min = my_acc(y_test, y_pred, sensitive_features=A_test["sex"])
print("Minimum accuracy_score: ", my_acc_min)

# %%
# To show that the returned function also works with sample weights:
random_weights = np.random.rand(len(y_test))

acc_frame_sw = MetricFrame(
    metrics=skm.accuracy_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
    sample_params={"sample_weight": random_weights},
)

from_frame = acc_frame_sw.group_min()
from_func = my_acc(
    y_test,
    y_pred,
    sensitive_features=A_test["sex"],
    sample_weight=random_weights,
)

print("From MetricFrame:", from_frame)
print("From function   :", from_func)

# %%
# The returned function can also handle parameters which are not sample
# parameters. Consider :func:`sklearn.metrics.fbeta_score`, which
# has a required :code:`beta=` argument (and suppose that this time
# we are most interested in the maximum difference to the overall value).
# First we evaluate this with a :class:`fairlearn.metrics.MetricFrame`:

fbeta_03 = functools.partial(skm.fbeta_score, beta=0.3)
fbeta_03.__name__ = "fbeta_score__beta_0.3"

beta_frame = MetricFrame(
    metrics=fbeta_03,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test["sex"],
    sample_params={"sample_weight": random_weights},
)
beta_from_frame = beta_frame.difference(method="to_overall")

print("From frame:", beta_from_frame)

# %%
# And next, we create a function to evaluate the same. Note that
# we do not need to use :func:`functools.partial` to bind the
# :code:`beta=` argument:

beta_func = make_derived_metric(metric=skm.fbeta_score, transform="difference")

beta_from_func = beta_func(
    y_test,
    y_pred,
    sensitive_features=A_test["sex"],
    beta=0.3,
    sample_weight=random_weights,
    method="to_overall",
)

print("From function:", beta_from_func)


# %%
# Pregenerated Metrics
# ====================
#
# We provide a number of pregenerated metrics, to cover
# common use cases. For example, we provide a
# :code:`accuracy_score_group_min()` function to
# find the minimum over the accuracy scores:


from_myacc = my_acc(y_test, y_pred, sensitive_features=A_test["race"])

from_pregen = accuracy_score_group_min(
    y_test, y_pred, sensitive_features=A_test["race"]
)

print("From my function :", from_myacc)
print("From pregenerated:", from_pregen)
assert from_myacc == from_pregen
