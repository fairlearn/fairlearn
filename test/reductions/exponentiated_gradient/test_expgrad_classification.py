# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions.moments import DemographicParity

from test.simple_learners import LeastSquaresBinaryClassifierLearner
from .test_utilities import group_data, X1, X2, X3, labels  # noqa: F401


def test_simple_fit_predict():
    estimator = LeastSquaresBinaryClassifierLearner()
    constraints = DemographicParity()
    expgrad = ExponentiatedGradient(estimator, constraints)
    expgrad.fit(pd.DataFrame(X1), pd.Series(labels), group_data=pd.Series(group_data))
    expgrad.predict(pd.DataFrame(X1))
