# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
================================================
Mitigating Fairness using Adversarial Mitigation
================================================
"""
# %%
# This notebook demonstrates our implementation of the technique *Mitigating*
# *Unwanted Biases with Adversarial Learning* as proposed by
# `Zhang et al. 2018 <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_.
#
# In short, the authors take classic supervised learning setting in which
# a predictor neural network is trained, and extend it with an adversarial
# network that aims to predict the sensitive feature. Then, they train the
# predictor not only to minimize its own loss, but also minimize the predictive
# ability of the adversarial.
#
# In short, we provide an implementation that supports:
#
# - Any predictor neural network implemented in either PyTorch or Tensorflow2
# - Classification or regression
# - Multiple sensitive features
# - Two fairness objectives: Demographic parity or Equality of Odds
#
# This implementation follows closely the API of an `Estimator` in :class:`sklearn`

# %%
# Example 1: Simple use case with UCI Adult Dataset
# =================================================
# Firstly, we cover a most basic application of adversarial mitigation. However,
# this starts by loading and preprocessing the dataset. 
# 
# For this example we choose the sex as the sensitive feature.

from fairlearn.datasets import fetch_adult

# Get dataset
X, y = fetch_adult(as_frame=True, return_X_y=True)

# Remove rows with NaNs. In general dropping NaNs is not statistically sound,
# but for this example we ignore that.
non_NaN_rows = ~X.isna().any(axis=1)

X = X[non_NaN_rows]
y = y[non_NaN_rows]

# Choose sensitive feature
sensitive_feature = X['sex']

# %% 
# Clearly, the UCI adult dataset can not be fed into a neural network (yet),
# as we have many columns that are not numerical in nature. To resolve this
# issues, we could for instance use one-hot-encodings to preprocess categorical
# columns. Additionally, let's preprocess the columns of number to a
# standardized range. For these tasks, we can use functionality from
# `sklearn.preprocessor`.
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from numpy import number
from pandas import Series

def transform(X):
    if isinstance(X, Series):  # make_column_transformer works with DataFrames
        X = X.to_frame()
    ct = make_column_transformer(
        (StandardScaler(),
         make_column_selector(dtype_include=number)),
        (OneHotEncoder(drop='if_binary', sparse=False),
         make_column_selector(dtype_include="category")))
    return ct.fit_transform(X)


X = transform(X)
y = transform(y)
sensitive_feature = transform(sensitive_feature)

# %%
# Now, we can use :class:`fairlearn.adversarial.AdversarialClassifier` to train on the
# UCI Adult dataset. As our predictor and adversary models, we use for
# simplicity the fairlearn built-in constructors for fully connected neural
# networks with sigmoid activations. We initialize neural network constructors
# by passing a list :math:`h_1, h_2, \dots` that indicate the number of nodes
# :math:`h_i` per hidden layer :math:`i`.
# 
# The specific fairness
# objective that we choose for this example is demographic parity, so we also
# set :code:`objective = "demographic_parity"`.

from fairlearn.adversarial import AdversarialClassifier

mitigator = AdversarialClassifier(
    predictor_model=[50, 20],
    adversary_model=[6, 6],
    constraints="demographic_parity"
)

# %% 
# Then, we can fit the data to our model. We generally follow sklearn API,
# but in this case we require some extra kwargs. In particular, we should
# specify the number of epochs, batch size, whether to shuffle the rows of data
# after every epoch, and optionally after how many seconds to show a progress
# update.

mitigator.fit(
    X,
    y,
    sensitive_features=sensitive_feature,
    epochs=100,
    batch_size=2**10,
    shuffle=True,
    progress_updates=5
)

# %% 
# Predict and evaluate. In particular, we trained the predictor for demographic
# parity, so we are not only interested in the accuracy, but also in the selection
# rate.

y_pred = mitigator.predict(X)

from sklearn.metrics import accuracy_score
from fairlearn.metrics import equalized_odds_difference, MetricFrame, \
    true_positive_rate, false_positive_rate, selection_rate
mf = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate},
    y_true=y,
    y_pred=y_pred,
    sensitive_features=sensitive_feature)

print(mf.by_group)

# %%
# We see that the results are not great. The accuracy is not optimal, and there
# remains demographic disparity. 
# Of course, the success of this method of mitigating fairness
# depends strongly on neural network models and the specific hyperparameters, so
# when applying this method one should pay close attention to their parameter
# settings.
