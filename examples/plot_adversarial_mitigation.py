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
# In short, the authors take the classic supervised learning setting in which
# a predictor neural network is trained, and extend it with an adversarial
# network that aims to predict the sensitive feature given these
# predictions and possibly also given the true target. Then, they train the
# predictor not only to minimize its own loss, but also minimize the predictive
# ability of the adversarial. If this model converges properly,
# the adversary will attain a loss equal to the entropy, so the adversary
# can not predict the sensitive features from the predictions.
#
# We provide an implementation that supports:
#
# - Any predictor neural network implemented in either PyTorch or Tensorflow2
# - Classification (binary and categorical) or (multidimensional) regression
# - Highly customizable parameters, allowing multi-output
# prediction models and sensitive features.
# - Two fairness objectives: Demographic parity or Equalized Odds
# - A scikit-learn compliant API


# %%
# Imports used by the rest of the script
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
)
from fairlearn.adversarial import AdversarialFairnessClassifier

from numpy import number, mean
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
from math import sqrt

# %%
# Example 1: Simple use case with UCI Adult Dataset
# =================================================
# Firstly, we cover the most basic application of adversarial mitigation.
# We start by loading and preprocessing the dataset.
#
# For this example we choose the feature 'sex' as the sensitive feature.
# %%
# Fetch the data
X, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
pos_label = y[0]

# Choose sensitive feature
z = X["sex"]

# %%
# The UCI adult dataset can not be fed into a neural network (yet),
# as we have many columns that are not numerical in nature. To resolve this
# issue, we could for instance use one-hot-encodings to preprocess categorical
# columns. Additionally, let's preprocess the columns of number to a
# standardized range. For these tasks, we can use functionality from
# `sklearn.preprocessor`. Also, if we don't want to blatantly transform
# NaN's (which are common in this dataset) to zero's, then we should take
# care and define our own missing data imputer.


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
                ("encoder", OneHotEncoder(drop="if_binary", sparse=False)),
            ]
        ),
        make_column_selector(dtype_include="category"),
    ),
)

X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
    X, y, z, test_size=0.2, random_state=12345, stratify=y
)

X_prep_train = ct.fit_transform(X_train)
X_prep_test = ct.transform(X_test)


# %%
# Now, we can use :class:`fairlearn.adversarial.AdversarialFairnessClassifier`
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
# update.


mitigator = AdversarialFairnessClassifier(
    backend="tensorflow",
    predictor_model=[50, "leaky_relu"],
    adversary_model=[3, "leaky_relu"],
    batch_size=2 ** 8,
    progress_updates=0.5,
    random_state=123,
)

# %%
# Then, we can fit the data to our model.

mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

# %%
# Predict and evaluate. In particular, we trained the predictor for demographic
# parity, so we are not only interested in the accuracy, but also
# in the selection
# rate.

predictions = mitigator.predict(X_prep_test)

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=Y_test == pos_label,
    y_pred=predictions == pos_label,
    sensitive_features=Z_test,
)

print(mf.by_group)

# %%
# When using adversarial debiasing out-of-the-box, users may not yield such
# good training results after the first attempt. In general, training
# adversarial networks is hard, and users may need to tweak the
# hyperparameters continuously. Besides general scikit-learn algorithms
# that finetune estimators,
# Example 2 will demonstrate some problem-specific
# techniques we can use such as using dynamic hyperparameters,
# validation, and early stopping to improve adversarial training.

# %%
# Example 2: Finetuning
# =====================
# Adversarial Learning is inherently difficult because of various issues,
# such as mode collapse, divergence, and diminishing gradients.
# In particular, mode collapse seems a real problem on this dataset: the
# predictor and adversary trap themselves in a local minimum by favoring one
# class (mode). Problems with diverging parameters may also occur, which
# may be an indication of a bad choice of hyperparameters, such as a
# learning rate that is too large. The problems teh user may encounter are
# of course case specific, but general good practices when training
# such models are: train slowly, ensuring the
# losses remain balanced, and keep track of validation accuracies.
# Additionally, we found that single hidden layer neural
# networks work best for this use case.

# In this example, we demonstrate some of these good practices.
# We start by defining our
# predictor neural network explicitely so that it is more apparant.
# We will be using PyTorch, but the same can be achieved using Tensorflow!


class PredictorModel(torch.nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(X_prep_train.shape[1], 200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(200, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


predictor_model = PredictorModel()
# %%
# We also take a look at
# some validation
# metrics. Most importantly, we chose the demographic parity difference
# to check to what
# extent the constraint (demographic parity in this case) is satisfied.
# We also look at the selection_rate to observe whether our model is
# suffering from mode collapse, and we also calculate the accuracy on the
# validation set as well.
# We will pass this validation step to our model later.


def validate(mitigator):
    predictions = mitigator.predict(X_prep_test)
    dp_diff = demographic_parity_difference(
        Y_test == pos_label, predictions == pos_label, sensitive_features=Z_test
    )
    accuracy = mean(predictions.values == Y_test.values)
    selection_rate = mean(predictions == pos_label)
    print(
        "DP diff: {:.4f}, accuracy: {:.4f}, selection_rate: {:.4f}".format(
            dp_diff, accuracy, selection_rate
        )
    )
    return dp_diff, accuracy, selection_rate


# %%
# We may define the optimizers however we like. In this case, we use the
# suggestion from the paper to set the hyperparameters alpha and learning
# rate (:math:`\mu`) to depend on the timestep such that :math:`\alpha \cdot \mu
# \rightarrow 0` as the timestep grows.

# We make use of a callback function to both update the hyperparameters and to
# validate the model. We update these hyperparameters at every 10 steps, and we
# validate every 100 steps. Additionally, we can implement early stopping
# easily by calling :code:`return True` in a callback function.


schedulers = []


def optimizer_constructor(model):
    global schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    schedulers.append(
        torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    )
    return optimizer


step = 1


def callbacks(model, *args):
    global step
    global schedulers
    step += 1
    # Update hyperparameters
    model.alpha = 0.3 * sqrt(step // 1)
    for scheduler in schedulers:
        scheduler.step()
    # Validate (and early stopping) every 50 steps
    if step % 50 == 0:
        dp_diff, accuracy, selection_rate = validate(model)
        # Early stopping condition:
        # Good accuracy + low dp_diff + no mode collapse
        if (
            dp_diff < 0.03
            and accuracy > 0.8
            and selection_rate > 0.01
            and selection_rate < 0.99
        ):
            return True


# %%
# Then, the instance itself. Notice that we do not explicitely define loss
# functions, because adversarial fairness is able to infer the loss function
# on its own in this example.

mitigator = AdversarialFairnessClassifier(
    predictor_model=predictor_model,
    adversary_model=[3, "leaky_relu"],
    predictor_optimizer=optimizer_constructor,
    adversary_optimizer=optimizer_constructor,
    epochs=10,
    batch_size=2 ** 7,
    shuffle=True,
    callbacks=callbacks,
    random_state=123,
)

# %%
# Finally, we fit the model

mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

validate(mitigator)

# %%
# We take a look at the results. Notice we achieve a much lower demographic
# parity
# difference than in Exercise 1! This does come at the cost of some accuracy,
# but
# such a tradeof is to be expected as we are purposely mitigating
# the unfairness that
# was present in the data.

predictions = mitigator.predict(X_prep_test)

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=Y_test == pos_label,
    y_pred=predictions == pos_label,
    sensitive_features=Z_test,
)

print(mf.by_group)

# %%
# Example 3: Scikit-learn applications
# ====================================
# AdversarialFairness is quite compliant with scikit-learn API, so functions
# such as pipelining and model selection are applicable here. In particular,
# applying pipelining might seem complicated as scikit-learn only pipelines
# :code:`X` and :code:`y``, not the :code:`sensitive_features`.
# We overcome this issue by passing the sensitive features through the
# pipeline as keyword-argument
# :code:`[name of model]__sensitive_features` to fit.

pipeline = Pipeline(
    [
        ("preprocessor", ct),
        (
            "classifier",
            AdversarialFairnessClassifier(
                backend="tensorflow",
                predictor_model=[50, "leaky_relu"],
                adversary_model=[3, "leaky_relu"],
                batch_size=2 ** 8,
                random_state=123,
            ),
        ),
    ]
)

pipeline.fit(X_train, Y_train, classifier__sensitive_features=Z_train)

predictions = pipeline.predict(X_test)

mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=Y_test == pos_label,
    y_pred=predictions == pos_label,
    sensitive_features=Z_test,
)

print(mf.by_group)

# %%
