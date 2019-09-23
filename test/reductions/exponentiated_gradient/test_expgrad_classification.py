# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions.moments import DemographicParity

from test.simple_learners import LeastSquaresBinaryClassifierLearner
from .test_utilities import aux_data, X1, X2, X3, labels


def test_simple_fit_predict():
    estimator = LeastSquaresBinaryClassifierLearner()
    disparity_metric = DemographicParity()
    # TODO what about the quality metric?
    expgrad = ExponentiatedGradient(estimator, disparity_metric)
    expgrad.fit(pd.DataFrame(X1), pd.Series(labels), aux_data=pd.Series(aux_data))
    expgrad.predict(pd.DataFrame(X1))
