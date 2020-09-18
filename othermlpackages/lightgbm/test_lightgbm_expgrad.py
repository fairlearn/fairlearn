# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from copy import deepcopy
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity

import lightgbm as lgb

def test_smoke():

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

    lgb_params = {
        'objective' : 'binary',
        'metric' : 'auc',
        'learning_rate': 0.03,
        'num_leaves' : 10,
        'max_depth' : 3
    }
    estimator = lgb.LGBMClassifier(**lgb_params)
    disparity_moment = DemographicParity()
    eps = 0.01

    expgrad = ExponentiatedGradient(
            estimator,
            constraints=deepcopy(disparity_moment),
            eps=eps)

    expgrad.fit(X_scaled, Y, sensitive_features=A)
    assert expgrad.n_oracle_calls_ > 1
    assert len(expgrad.predictors_) > 1