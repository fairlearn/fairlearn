# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from fairlearn.reductions import ExponentiatedGradient, GridSearch, DemographicParity, \
    EqualizedOdds


@pytest.mark.parametrize("Mitigator", [ExponentiatedGradient, GridSearch])
@pytest.mark.parametrize("Constraints", [DemographicParity, EqualizedOdds])
@pytest.mark.parametrize("Estimator", [LGBMClassifier, LogisticRegression, SVC,
                                       DecisionTreeClassifier])
def test_smoke(Mitigator, Constraints, Estimator):
    # This test case ensures that input validation doesn't remove metadata from the input
    # matrix X, as described at https://github.com/fairlearn/fairlearn/issues/312
    np.random.seed(0)
    n = 100
    X0 = np.random.normal(size=n)
    X1 = np.random.choice([1, 2, 3], size=n)
    Y = np.random.choice([0, 1], size=n)
    A = np.random.choice([0, 1], size=n)
    df = pd.DataFrame({"X0": X0, "X1": X1})
    # Set X1 as categorical
    df['X1'] = df['X1'].astype('category')

    mitigator = Mitigator(Estimator(), Constraints())
    mitigator.fit(df, Y, sensitive_features=A)
    mitigator.predict(df)
