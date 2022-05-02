# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.utils.estimator_checks import parametrize_with_checks

from fairlearn.preprocessing import CorrelationRemover


@parametrize_with_checks(
    [
        CorrelationRemover(sensitive_feature_ids=[]),
        CorrelationRemover(sensitive_feature_ids=[0]),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


def test_linear_dependence():
    X = np.array(
        [
            [
                0,
                0,
                1,
                1,
            ],
            [
                1,
                1,
                2,
                2,
            ],
            [
                0.1,
                0.2,
                1.2,
                1.1,
            ],
        ]
    ).T

    X_tfm = CorrelationRemover(sensitive_feature_ids=[0]).fit(X).transform(X)
    assert X_tfm.shape[1] == 2
    assert np.allclose(X_tfm[:, 0], 1.5)


def test_linear_dependence_pd():
    X = np.array(
        [
            [
                0,
                0,
                1,
                1,
            ],
            [
                1,
                1,
                2,
                2,
            ],
            [
                0.1,
                0.2,
                1.2,
                1.1,
            ],
        ]
    ).T

    df = pd.DataFrame(X, columns=["a", "b", "c"])

    X_tfm = CorrelationRemover(sensitive_feature_ids=["a"]).fit(df).transform(df)
    assert X_tfm.shape[1] == 2
    assert np.allclose(X_tfm[:, 0], 1.5)
