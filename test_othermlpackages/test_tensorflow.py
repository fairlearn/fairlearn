# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest
import sklearn
from packaging.version import parse

from fairlearn.adversarial import AdversarialFairnessClassifier
from fairlearn.reductions import DemographicParity

from . import package_test_common as ptc

tf = pytest.importorskip("tensorflow")
from keras.layers import Dense  # noqa
from keras.models import Sequential  # noqa
from scikeras.wrappers import KerasClassifier  # noqa


def _should_skip_test():
    return parse(sklearn.__version__) >= parse("1.6.0")


def create_model():
    # create model
    model = Sequential()
    # 103 is the number of X columns after the get_dummies() call
    model.add(Dense(12, input_dim=103, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_expgrad_classification():
    estimator = KerasClassifier(model=create_model)
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_gridsearch_classification():
    estimator = KerasClassifier(model=create_model)
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)


@pytest.mark.skipif(
    _should_skip_test(),
    reason="Skipped because of scikit-learn >= 1.6. Will be enabled again when the issues in the external library are fixed.",
)
def test_thresholdoptimizer_classification():
    estimator = KerasClassifier(model=create_model)

    ptc.run_thresholdoptimizer_classification(estimator)


def test_adversarial_classification():
    mitigator = AdversarialFairnessClassifier(
        backend="tensorflow",
        predictor_model=[50, "relu"],
        adversary_model=[3, "relu"],
        random_state=123,
    )

    ptc.run_AdversarialFairness_classification(mitigator)
