# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from . import package_test_common as ptc

from fairlearn.reductions import DemographicParity

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def test_expgrad_classification():
    estimator = KerasClassifier(build_fn=create_model)
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)
