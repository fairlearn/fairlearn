"""
.. _adversarial_Example_1:

===============================================================
Basics & Model Specification of `AdversarialFairnessClassifier`
===============================================================

This example shows how to use
:class:`~fairlearn.adversarial.AdversarialFairnessClassifier` on the UCI Adult
dataset.
"""

# %%
# First, we cover a most basic application of adversarial mitigation.
# We start by loading and preprocessing the dataset:

from fairlearn.datasets import fetch_adult

X, y = fetch_adult(return_X_y=True)
pos_label = y[0]

z = X["sex"]  # In this example, we consider 'sex' the sensitive feature.

# %%
# As with other machine learning methods, it is wise to take a train-test split
# of the data in order to validate the model on unseen data:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, Z_train, Z_test = train_test_split(
    X, y, z, test_size=0.2, random_state=12345, stratify=y
)

# %%
# The UCI adult dataset cannot be fed into a neural network (yet),
# as we have many columns that are not numerical in nature. To resolve this
# issue, we could for instance use one-hot encodings to preprocess categorical
# columns. Additionally, let's preprocess the numeric columns to a
# standardized range. For these tasks, we can use functionality from
# scikit-learn (:py:mod:`sklearn.preprocessing`). We also use an imputer
# to get rid of NaN's:

import sklearn
from numpy import number
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sklearn.set_config(enable_metadata_routing=True)

ct = make_column_transformer(
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("normalizer", StandardScaler()),
            ]
        ),
        make_column_selector(dtype_include=number),
    ),
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False)),
            ]
        ),
        make_column_selector(dtype_include="category"),
    ),
)

# %%
# Now, we can use :class:`~fairlearn.adversarial.AdversarialFairnessClassifier`
# to train on the
# UCI Adult dataset. As our predictor and adversary models, we use for
# simplicity the default constructors for fully connected neural
# networks with sigmoid activations implemented in Fairlearn. We initialize
# neural network constructors
# by passing a list :math:`h_1, h_2, \dots` that indicate the number of nodes
# :math:`h_i` per hidden layer :math:`i`. You can also put strings in this list
# to indicate certain activation functions, or just pass an initialized
# activation function directly.
#
# The specific fairness
# objective that we choose for this example is demographic parity, so we also
# set :code:`objective = "demographic_parity"`. We generally follow sklearn API,
# but in this case we require some extra kwargs. In particular, we should
# specify the number of epochs, batch size, whether to shuffle the rows of data
# after every epoch, and optionally after how many seconds to show a progress
# update:

from fairlearn.adversarial import AdversarialFairnessClassifier

mitigator = AdversarialFairnessClassifier(
    backend="torch",
    predictor_model=[50, "leaky_relu"],
    adversary_model=[3, "leaky_relu"],
    batch_size=2**8,
    progress_updates=0.5,
    random_state=123,
)

# %%
# We now put the above model in a ``Pipeline`` with the transformation step. Note
# that we use ``scikit-learn``'s metadata routing to pass the sensitive feature::

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(ct, mitigator.set_fit_request(sensitive_features=True))

# %%
# Then, we can fit the data to our model:

pipeline.fit(X_train, y_train, sensitive_features=Z_train)

# %%
# Finally, we evaluate the predictions. In particular, we trained the
# predictor for demographic parity, so we are not only interested in
# the accuracy, but also in the selection rate. MetricFrames are a great resource
# here:

predictions = pipeline.predict(X_test)

from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, selection_rate

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test == pos_label,
    y_pred=predictions == pos_label,
    sensitive_features=Z_test,
)

# %%
# Then, to display the result:

print(mf.by_group)

# The above statistics tell us that the accuracy of our model is quite good,
# 90% for females and 72% for males. However, the selection rates differ, so there
# is a large demographic disparity here. When using adversarial fairness
# out-of-the-box, users may not yield such
# good training results after the first attempt. In general, training
# adversarial networks is hard, and users may need to tweak the
# hyperparameters continuously. Besides general scikit-learn algorithms
# that finetune estimators,
# :ref:`adversarial_Example_2` will demonstrate some problem-specific
# techniques we can use such as using dynamic hyperparameters,
# validation, and early stopping to improve adversarial training.
