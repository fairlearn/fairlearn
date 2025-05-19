# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest
from keras.src.layers import Dense, Input
from keras.src.models import Model
from keras.wrappers import SKLearnClassifier

from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.reductions import DemographicParity

from . import package_test_common as ptc

tf = pytest.importorskip("tensorflow")


def create_model(X, y, loss="binary_crossentropy", layers=[12, 8]):
    # create model
    n_features_in = X.shape[1]
    inp = Input(shape=(n_features_in,))

    hidden = inp
    for layer_size in layers:
        hidden = Dense(layer_size, activation="relu")(hidden)

    n_outputs = y.shape[1] if len(y.shape) > 1 else 1
    out = [Dense(n_outputs, activation="sigmoid")(hidden)]
    model = Model(inp, out)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    return model


def test_expgrad_classification():
    estimator = SKLearnClassifier(model=create_model)
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


def test_gridsearch_classification():
    estimator = SKLearnClassifier(model=create_model)
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)


def test_thresholdoptimizer_classification():
    estimator = SKLearnClassifier(model=create_model)

    ptc.run_thresholdoptimizer_classification(estimator)


def test_adversarial_classification():
    mitigator = AdversarialFairnessClassifier(
        backend="tensorflow",
        predictor_model=[50, "relu"],
        adversary_model=[3, "relu"],
        random_state=123,
    )

    ptc.run_AdversarialFairness_classification(mitigator)
