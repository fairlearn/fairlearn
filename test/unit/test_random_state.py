# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds


def test_random_state_threshold_optimizer():
    """Test that the random_state argument works as expected.

    This test case reproduces the problem reported in issue 588 if the
    random_state does not work as intended within ThresholdOptimizer.
    https://github.com/fairlearn/fairlearn/issues/588
    """
    X_train, X_test, y_train, y_test, race_train, race_test = _get_test_data()

    # Train a simple logistic regression model
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_train, y_train)

    # Train threshold optimizer
    to = ThresholdOptimizer(estimator=lr, constraints='equalized_odds', grid_size=1000)
    to.fit(X_train, y_train, sensitive_features=race_train)

    # score groups
    y_pred_test = to.predict(X_test, sensitive_features=race_test, random_state=0)
    for _ in range(100):
        assert (y_pred_test ==
                to.predict(X_test, sensitive_features=race_test, random_state=0)).all()
    assert (y_pred_test != to.predict(X_test, sensitive_features=race_test, random_state=1)).any()


def test_random_state_exponentiated_gradient():
    """Test that the random_state argument works as expected.

    This test case reproduces the problem reported in issue 588 if the
    random_state does not work as intended within Exponentiated Gradient.
    https://github.com/fairlearn/fairlearn/issues/588
    """
    X_train, X_test, y_train, y_test, race_train, race_test = _get_test_data()

    # Train a simple logistic regression model
    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_train, y_train)

    # Train threshold optimizer
    expgrad = ExponentiatedGradient(estimator=lr, constraints=EqualizedOdds())
    expgrad.fit(X_train, y_train, sensitive_features=race_train)

    # score groups
    y_pred_test = expgrad.predict(X_test, random_state=0)
    for _ in range(100):
        assert (y_pred_test ==
                expgrad.predict(X_test, random_state=0)).all()
    assert (y_pred_test !=
            expgrad.predict(X_test, random_state=1)).any()


def _get_test_data():
    # fetch data from OpenML
    data = fetch_openml(data_id=42193)
    X = pd.DataFrame(data['data'], columns=data['feature_names']) \
        .drop(columns=['race_Caucasian', 'c_charge_degree_F'])
    y = data['target'].astype(np.int)

    # split the data in train-validation-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    race_train = X_train['race_African-American']
    race_test = X_test['race_African-American']
    return X_train, X_test, y_train, y_test, race_train, race_test
