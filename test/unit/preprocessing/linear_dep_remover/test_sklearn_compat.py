# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest
import numpy as np
import pandas as pd
from sklearn.utils import estimator_checks

from fairlearn.preprocessing import CorrelationRemover


@pytest.mark.parametrize(
    "test_fn",
    [
        # transformer checks
        estimator_checks.check_transformer_general,
        estimator_checks.check_transformers_unfitted,
        # general estimator checks
        estimator_checks.check_fit2d_predict1d,
        estimator_checks.check_methods_subset_invariance,
        estimator_checks.check_fit2d_1sample,
        estimator_checks.check_fit2d_1feature,
        estimator_checks.check_get_params_invariance,
        estimator_checks.check_set_params,
        estimator_checks.check_dict_unchanged,
        estimator_checks.check_dont_overwrite_parameters,
        # nonmeta_checks
        estimator_checks.check_estimators_dtypes,
        estimator_checks.check_fit_score_takes_y,
        estimator_checks.check_dtype_object,
        estimator_checks.check_sample_weights_pandas_series,
        estimator_checks.check_sample_weights_list,
        estimator_checks.check_estimators_fit_returns_self,
        estimator_checks.check_complex_data,
        estimator_checks.check_estimators_empty_data_messages,
        estimator_checks.check_pipeline_consistency,
        estimator_checks.check_estimators_nan_inf,
        estimator_checks.check_estimators_overwrite_params,
        estimator_checks.check_estimator_sparse_data,
        estimator_checks.check_estimators_pickle,
    ]
)
def test_estimator_checks(test_fn):
    test_fn(CorrelationRemover.__name__, CorrelationRemover(sensitive_feature_ids=[]))
    test_fn(CorrelationRemover.__name__, CorrelationRemover(sensitive_feature_ids=[0]))


def test_linear_dependence():
    X = np.array([[0, 0, 1, 1, ],
                  [1, 1, 2, 2, ],
                  [0.1, 0.2, 1.2, 1.1, ]]).T

    X_tfm = CorrelationRemover(sensitive_feature_ids=[0]).fit(X).transform(X)
    assert X_tfm.shape[1] == 2
    assert np.allclose(X_tfm[:, 0], 1.5)


def test_linear_dependence_pd():
    X = np.array([[0, 0, 1, 1, ],
                  [1, 1, 2, 2, ],
                  [0.1, 0.2, 1.2, 1.1, ]]).T

    df = pd.DataFrame(X, columns=['a', 'b', 'c'])

    X_tfm = CorrelationRemover(sensitive_feature_ids=['a']).fit(df).transform(df)
    assert X_tfm.shape[1] == 2
    assert np.allclose(X_tfm[:, 0], 1.5)
