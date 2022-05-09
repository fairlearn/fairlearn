# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from test.unit.input_convertors import (
    _map_into_single_column,
    conversions_for_1d,
    ensure_dataframe,
    ensure_ndarray,
)
from test.unit.reductions.conftest import is_invalid_transformation

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fairlearn.reductions import DemographicParity, ErrorRate, ExponentiatedGradient
from fairlearn.utils._input_validation import _LABELS_NOT_0_1_ERROR_MESSAGE

from .simple_learners import LeastSquaresBinaryClassifierLearner
from .test_utilities import _get_data

# ===============================================================

# Ways of transforming the data
candidate_X_transforms = [ensure_ndarray, ensure_dataframe]
candidate_Y_transforms = conversions_for_1d
candidate_A_transforms = conversions_for_1d

# ================================================================

_PRECISION = 1e-6


class TestExponentiatedGradientArguments:
    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_argument_types_difference_bound(
        self, transformX, transformY, transformA, A_two_dim
    ):
        # This is an expanded-out version of one of the smoke tests
        X, y, A = _get_data(A_two_dim)
        merged_A = _map_into_single_column(A)

        transformed_X = transformX(X)
        transformed_y = transformY(y)
        transformed_A = transformA(A)
        eps = 0.1

        expgrad = ExponentiatedGradient(
            LeastSquaresBinaryClassifierLearner(),
            constraints=DemographicParity(difference_bound=eps),
            eps=eps,
        )
        expgrad.fit(transformed_X, transformed_y, sensitive_features=transformed_A)

        def Q(X):
            return expgrad._pmf_predict(X)[:, 1]

        n_predictors = len(expgrad.predictors_)

        disparity_moment = DemographicParity(difference_bound=eps)
        disparity_moment.load_data(X, y, sensitive_features=merged_A)
        error = ErrorRate()
        error.load_data(X, y, sensitive_features=merged_A)
        disparity = disparity_moment.gamma(Q).max()
        disp = disparity_moment.gamma(Q)
        disp_eps = disparity_moment.gamma(Q) - disparity_moment.bound()
        error = error.gamma(Q)[0]

        assert expgrad.best_gap_ == pytest.approx(0.0000, abs=_PRECISION)
        assert expgrad.last_iter_ == 5
        assert expgrad.best_iter_ == 5
        assert disparity == pytest.approx(0.1, abs=_PRECISION)
        assert np.all(np.isclose(disp - eps, disp_eps))
        assert error == pytest.approx(0.25, abs=_PRECISION)
        assert expgrad.n_oracle_calls_ == 32
        assert n_predictors == 3

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_argument_types_ratio_bound(
        self, transformX, transformY, transformA, A_two_dim
    ):
        # This is an expanded-out version of one of the smoke tests
        X, y, A = _get_data(A_two_dim)
        merged_A = _map_into_single_column(A)

        transformed_X = transformX(X)
        transformed_y = transformY(y)
        transformed_A = transformA(A)
        eps = 0.1
        ratio = 1.0

        expgrad = ExponentiatedGradient(
            LeastSquaresBinaryClassifierLearner(),
            constraints=DemographicParity(ratio_bound_slack=eps, ratio_bound=ratio),
            eps=eps,
        )
        expgrad.fit(transformed_X, transformed_y, sensitive_features=transformed_A)

        def Q(X):
            return expgrad._pmf_predict(X)[:, 1]

        n_predictors = len(expgrad.predictors_)

        disparity_moment = DemographicParity(ratio_bound_slack=eps, ratio_bound=ratio)
        disparity_moment.load_data(X, y, sensitive_features=merged_A)
        error = ErrorRate()
        error.load_data(X, y, sensitive_features=merged_A)
        disparity = disparity_moment.gamma(Q).max()
        disp = disparity_moment.gamma(Q)
        disp_eps = disparity_moment.gamma(Q) - disparity_moment.bound()
        error = error.gamma(Q)[0]

        assert expgrad.best_gap_ == pytest.approx(0.0000, abs=_PRECISION)
        assert expgrad.last_iter_ == 5
        assert expgrad.best_iter_ == 5
        assert disparity == pytest.approx(0.1, abs=_PRECISION)
        assert np.all(np.isclose(disp - eps, disp_eps))
        assert error == pytest.approx(0.25, abs=_PRECISION)
        assert expgrad.n_oracle_calls_ == 32
        assert n_predictors == 3

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_input_X_unchanged(self, transformA, transformY, transformX, mocker):
        # The purpose of this test is to ensure that X is passed to the underlying estimator
        # unchanged. For y and sample_weight ExponentiatedGradient makes certain transformations
        # which are required. They are expected as pandas.Series.
        X, y, A = _get_data()

        transformed_X = transformX(X)
        transformed_y = transformY(y)
        transformed_A = transformA(A)

        # Using a mocked estimator here since we don't actually want to fit one, but rather care
        # about having that object's fit method called exactly twice through the best_h calls.
        estimator = LeastSquaresBinaryClassifierLearner()
        estimator.predict = mocker.Mock(return_value=y)
        estimator.fit = mocker.MagicMock()
        # restrict ExponentiatedGradient to a single iteration
        expgrad = ExponentiatedGradient(
            estimator, constraints=DemographicParity(), max_iter=1
        )
        mocker.patch("copy.deepcopy", return_value=estimator)
        expgrad.fit(transformed_X, transformed_y, sensitive_features=transformed_A)

        # ensure that the input data wasn't changed by our mitigator before being passed to the
        # underlying estimator
        assert estimator.fit.call_count == 2
        args, kwargs = estimator.fit.call_args
        assert len(args) == 2  # X and y
        assert len(kwargs) == 1  # sample_weight
        assert isinstance(args[0], type(transformed_X))
        assert isinstance(args[1], pd.Series)
        assert isinstance(kwargs["sample_weight"], pd.Series)

    def test_binary_classifier_0_1_required(self):
        X, y, A = _get_data()
        y = 2 * y

        expgrad = ExponentiatedGradient(
            LogisticRegression(), constraints=DemographicParity(), max_iter=1
        )
        with pytest.raises(ValueError) as execInfo:
            expgrad.fit(X, y, sensitive_features=(A))
        assert _LABELS_NOT_0_1_ERROR_MESSAGE == execInfo.value.args[0]

    def test_sample_weights_argument(self):
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(solver="liblinear")),
            ]
        )

        X, y, A = _get_data()

        expgrad = ExponentiatedGradient(
            estimator, constraints=DemographicParity(), max_iter=1
        )

        with pytest.raises(ValueError) as execInfo:
            expgrad.fit(X, y, sensitive_features=(A))
        assert (
            "Pipeline.fit does not accept the sample_weight parameter"
            in execInfo.value.args[0]
        )

        expgrad = ExponentiatedGradient(
            estimator,
            constraints=DemographicParity(),
            max_iter=1,
            sample_weight_name="logistic__sample_weight",
        )
        expgrad.fit(X, y, sensitive_features=(A))
