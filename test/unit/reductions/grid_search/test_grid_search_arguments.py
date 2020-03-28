# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.exceptions import NotFittedError
from fairlearn import _NO_PREDICT_BEFORE_FIT
from fairlearn._input_validation import \
    (_MESSAGE_Y_NONE,
     _SENSITIVE_FEATURES_NON_BINARY_ERROR_MESSAGE,
     _LABELS_NOT_0_1_ERROR_MESSAGE)
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, EqualizedOdds
from fairlearn.reductions import GroupLossMoment, ZeroOneLoss

from test.unit.input_convertors import conversions_for_1d, ensure_ndarray, ensure_dataframe
from test.unit.reductions.conftest import is_invalid_transformation
from test.unit.reductions.grid_search.utilities import assert_n_grid_search_results, _quick_data

# ==============================================================

# List the different datatypes which need to succeed for
# all GridSearch calls
candidate_X_transforms = [ensure_ndarray, ensure_dataframe]
candidate_Y_transforms = conversions_for_1d
candidate_A_transforms = conversions_for_1d


# Base class for tests
# Tests which must be passed by all calls to the GridSearch
# go here
class ArgumentTests:
    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_valid_inputs(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion, grid_size=2)
        X, Y, A = _quick_data(A_two_dim)
        gs.fit(transformX(X),
               transformY(Y),
               sensitive_features=transformA(A))
        assert_n_grid_search_results(2, gs)

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_X_is_None(self, transformY, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion, grid_size=3)
        _, Y, A = _quick_data(A_two_dim)

        with pytest.raises(ValueError) as execInfo:
            gs.fit(None,
                   transformY(Y),
                   sensitive_features=transformA(A))

        assert "Expected 2D array, got scalar array instead" in execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_is_None(self, transformX, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, _, A = _quick_data()

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   None,
                   sensitive_features=transformA(A))

        assert _MESSAGE_Y_NONE == execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_X_Y_different_rows(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, _, A = _quick_data()
        Y = np.random.randint(2, size=len(A)+1)

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   sensitive_features=transformA(A))

        expected_exception_message = "Found input variables with inconsistent numbers of samples"
        assert expected_exception_message in execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_X_A_different_rows(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, Y, _ = _quick_data(A_two_dim)
        A = np.random.randint(2, size=len(Y)+1)
        if A_two_dim:
            A = np.stack((A, A), -1)

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   sensitive_features=transformA(A))

        expected_exception_message = "Found input variables with inconsistent numbers of samples"
        assert expected_exception_message in execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_sensitive_feature_non_binary(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, Y, A = _quick_data(A_two_dim)

        if A_two_dim:
            A[0][0] = 0
            A[0][1] = 0
            A[1][0] = 1
            A[1][1] = 1
            A[2][0] = 2
            A[2][1] = 2
        else:
            A[0] = 0
            A[1] = 1
            A[2] = 2

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   sensitive_features=transformA(A))

        assert _SENSITIVE_FEATURES_NON_BINARY_ERROR_MESSAGE == execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_df_bad_columns(self, transformX, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, Y, A = _quick_data(A_two_dim)

        Y_two_col_df = pd.DataFrame({"a": Y, "b": Y})
        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   Y_two_col_df,
                   sensitive_features=transformA(A))

        assert "bad input shape" in execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_ndarray_bad_columns(self, transformX, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, Y, A = _quick_data(A_two_dim)

        Y_two_col_ndarray = np.stack((Y, Y), -1)
        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   Y_two_col_ndarray,
                   sensitive_features=transformA(A))

        assert "bad input shape" in execInfo.value.args[0]

    # ----------------------------

    def test_no_predict_before_fit(self):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, _, _ = _quick_data()

        with pytest.raises(NotFittedError) as execInfo:
            gs.predict(X)

        assert _NO_PREDICT_BEFORE_FIT == execInfo.value.args[0]

    def test_no_predict_proba_before_fit(self):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, _, _ = _quick_data()

        with pytest.raises(NotFittedError) as execInfo:
            gs.predict_proba(X)

        assert _NO_PREDICT_BEFORE_FIT == execInfo.value.args[0]


# Tests specific to Classification
class ConditionalOpportunityTests(ArgumentTests):
    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_ternary(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, Y, A = _quick_data(A_two_dim)
        Y[0] = 0
        Y[1] = 1
        Y[2] = 2

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   sensitive_features=transformA(A))

        assert _LABELS_NOT_0_1_ERROR_MESSAGE == execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_not_0_1(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(self.estimator, self.disparity_criterion)
        X, Y, A = _quick_data(A_two_dim)
        Y = Y + 1

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   sensitive_features=transformA(A))

        assert _LABELS_NOT_0_1_ERROR_MESSAGE == execInfo.value.args[0]


# Set up DemographicParity
class TestDemographicParity(ConditionalOpportunityTests):
    def setup_method(self, method):
        self.estimator = LogisticRegression(solver='liblinear')
        self.disparity_criterion = DemographicParity()


# Test EqualizedOdds
class TestEqualizedOdds(ConditionalOpportunityTests):
    def setup_method(self, method):
        self.estimator = LogisticRegression(solver='liblinear')
        self.disparity_criterion = EqualizedOdds()


# Tests specific to BoundedGroupLoss
class TestBoundedGroupLoss(ArgumentTests):
    def setup_method(self, method):
        self.estimator = LinearRegression()
        self.disparity_criterion = GroupLossMoment(ZeroOneLoss())
