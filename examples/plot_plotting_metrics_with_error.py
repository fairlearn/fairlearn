# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
============================
Plotting Metrics with Errors
============================
"""
# %%
# Load and preprocess the data set
# ================================
# We start by importing the various modules we're going to use:
from fairlearn.experimental.enable_metric_frame_plotting import plot_metric_frame
from fairlearn.metrics import MetricFrame
from sklearn.datasets import fetch_openml
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
import numpy as np

# %%
# We download the data set using :meth:`fetch_openml` function in :class:`sklearn.datasets`.
# The original Adult data set can be found at https://archive.ics.uci.edu/ml/datasets/Adult
# There are some caveats to using this dataset, but we will use it solely as an example
# to demonstrate the plotting metrics with error bars functionality

data = fetch_openml(data_id=1590, as_frame=True)
X = data.data
y = (data.target == '>50K') * 1
X_train, X_test, y_train_true, y_test_true = train_test_split(
    X, y, test_size=0.33, random_state=42)

numeric_transformer = Pipeline(
    steps=[
        ("impute", SimpleImputer()),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
test_set_sex = X_test['race']

# %%
# Classifier
# ==========
# Now we use a :py:class:`sklearn.tree.DecisionTreeClassifier` to make predictions
#

classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X_train_processed, y_train_true)
y_test_pred = classifier.predict(X_test_processed)

# %%
# Error metric calculations
# =============
# We have many different choices for error metrics. In this notebook we'll just be using a
# `Normal approximation interval <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval>`_ (symmetric)
# and a `Wilson score interval <https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval>`_ (asymmetric).

# We aim to create a 95% confidence interval, so we use a :code:`z_score` of 1.959964
z_score = 1.959964
digits_of_precision = 4


def general_wilson(p, n, digits=digits_of_precision, z=z_score):
    """Return lower and upper bound using Wilson Interval."""
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = np.sqrt((p*(1 - p) + z*z / (4*n)))/np.sqrt(n)
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    return [round(lower_bound, digits), round(upper_bound, digits)]


def general_normal_err_binomial(p, n, digits=digits_of_precision, z=z_score):
    """Return standard error (for binary classification).

    Assumes infinitely large population.
    Should be used when the sampling fraction is small.
    For sampling fraction > 5%, may want to use finite population correction [1]

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Margin_of_error
    """
    return round(z*np.sqrt(p*(1.0-p))/np.sqrt(n), digits)


def recall_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = general_wilson(tp/(tp+fn), tp + fn, digits_of_precision, z_score)
    return bounds


def recall_normal_err(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    error = general_normal_err_binomial(tp/(tp+fn), tp + fn, digits_of_precision, z_score)
    return error


def accuracy_wilson(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    bounds = general_wilson(score, len(y_true), digits_of_precision, z_score)
    return bounds


def accuracy_normal_err(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    score = accuracy_score(y_true, y_pred)
    error = general_normal_err_binomial(score, len(y_true), digits_of_precision, z_score)
    return error


# %%
# MetricFrame
# ===========
# Now we create a :class:`fairlearn.metrics.MetricFrame` to generate the recall Wilson bounds and recall normal error
#

# Analyze metrics using :class:`fairlearn.metrics.MetricFrame`
metrics_dict = {
    'Recall': recall_score,
    'Recall Bounds': recall_wilson,
    'Recall Error': recall_normal_err,
    'Accuracy': accuracy_score,
    'Accuracy Bounds': accuracy_wilson,
    'Accuracy Error': accuracy_normal_err,
}
metric_frame = MetricFrame(metrics=metrics_dict, y_true=y_test_true, y_pred=y_test_pred, sensitive_features=test_set_sex)

# %%
# Plotting
# ========
# Plot metrics without error bars
# -------------------------------
plot_metric_frame(metric_frame, kind="scatter", metrics=['Recall', 'Accuracy'])

# %%
# Plot metrics with (symmetric) error bars
# ----------------------------------------
plot_metric_frame(metric_frame, kind="scatter",
                  metrics=['Recall', 'Accuracy'],
                  errors=['Recall Error', 'Accuracy Error'],
                  figsize=(12, 4), subplots=True, layout=[2, 1])
plot_metric_frame(metric_frame, kind="bar", metrics='Recall', errors='Recall Error', colormap="Pastel1")


# %%
# Plot metrics with confidence intervals (possibly asymmetric)
# ------------------------------------------------------------
plot_metric_frame(metric_frame, kind="bar", metrics=[
                  'Recall', 'Accuracy'], conf_intervals=['Recall Bounds', 'Accuracy Bounds'], plot_error_labels=True, subplots=False)
plot_metric_frame(metric_frame, kind="scatter",
                  metrics='Recall', conf_intervals='Recall Bounds')

# %%
# Plot metrics with error labels
# ------------------------------
plot_metric_frame(metric_frame, kind="bar", metrics='Recall', errors='Recall Error', colormap="Pastel1", plot_error_labels=True)

# %%
# Plots all columns and treats them as metrics without error bars
# ---------------------------------------------------------------
plot_metric_frame(metric_frame, kind="bar", colormap="rainbow", layout=[1, 2])

# %%
# Customizing plots
# -----------------
# :func:`plot_metric_frame` returns an :class:`Axes` object that we can customize futher.

axs = plot_metric_frame(metric_frame,
                        metrics=['Recall', 'Accuracy'],
                        errors=['Recall Error', 'Accuracy Error'],
                        figsize=(12, 4),
                        kind="scatter",
                        colormap="rainbow",
                        rot=45,
                        subplots=True,
                        layout=[1, 2],
                        legend_label="95% Confidence Interval")
axs[0][0].set_ylabel("Recall")
axs[0][0].set_title("Recall Plot")
axs[0][1].set_title("Accuracy Plot")
axs[0][0].set_xlabel("Race")
axs[0][1].set_xlabel("Race")
axs[0][0].set_ylabel("Recall")
axs[0][1].set_ylabel("Accuracy")

# Set the y-scale for both metrics to [0, 1]
axs[0][0].set_ylim((0, 1))
axs[0][1].set_ylim((0, 1))

# %%
# Defining custom error metrics
# =============================
# In this tutorial we used Normal intervals and Wilson bounds to demonstrate how to plot symmetric and asymmetric metrics, respectively.
#
# However, in order to support many different metrics, we reuse the :class:`fairlearn.metrics.MetricFrame` functionality by writing a function
# that takes in :code:`y_true` and :code:`y_pred`, and returns the error metric:


def custom_error_metric_function(y_true, y_pred):
    assert len(y_true) == len(y_pred)

    # compute custom metric function here
    # In this case, we will calculate a wilson bound for recall
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    bounds = general_wilson(tp/(tp+fn), tp + fn, digits_of_precision,
                            z_score)

    # returns the bounds in the format (lower_bound, upper_bound)
    return bounds


# %%
metrics_dict = {
    'Recall': recall_score,
    'Recall Bounds': custom_error_metric_function,
}
metric_frame = MetricFrame(metrics=metrics_dict, y_true=y_test_true, y_pred=y_test_pred, sensitive_features=test_set_sex)

# %%
# Plotting
# ========
# Plot metrics without error bars
# -------------------------------
plot_metric_frame(metric_frame, kind="bar", metrics='Recall', conf_intervals='Recall Bounds')

# %%
