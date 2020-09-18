# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from copy import deepcopy
import numpy as np
import pandas as pd
import pytest

from fairlearn.datasets import fetch_adult
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity

import lightgbm as lgb

def test_smoke():
    data = fetch_adult(as_frame=True)
    X = data.data
    y_true = (data.target == '>50K') * 1
    sex = X['sex']

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

    expgrad.fit(X, y_true, sensitive_features=sex)
    assert expgrad.n_oracle_calls_ > 1
    assert len(expgrad.predictors_) > 1