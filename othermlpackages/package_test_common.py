# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.reductions import ExponentiatedGradient, GridSearch

def fetch_adult():
    data = fetch_openml(data_id=1590, as_frame=True)
    X = data.data.drop(labels=['sex'], axis=1)
    X = pd.get_dummies(X)
    Y = (data.target == '>50K') * 1
    A = data.data['sex']
    A = pd.get_dummies(A)
    
    le = LabelEncoder()
    Y = le.fit_transform(Y)

    le = LabelEncoder()
    

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, Y, A


def run_expgrad_classification(estimator, moment):
    X, Y, A = fetch_adult()

    expgrad = ExponentiatedGradient(
            estimator,
            constraints=moment)
    expgrad.fit(X, Y, sensitive_features=A)

    assert expgrad.n_oracle_calls_ > 1
    assert len(expgrad.predictors_) > 1

def run_gridsearch_classification(estimator, moment):
    X, Y, A = fetch_adult()

    num_predictors = 5
    gs = GridSearch(
            estimator,
            constraints=moment,
            grid_size=num_predictors)
    gs.fit(X, Y, sensitive_features=A)

    assert len(gs.predictors_) == num_predictors
