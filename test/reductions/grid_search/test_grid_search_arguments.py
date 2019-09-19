# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from fairlearn.metrics import DemographicParity, BoundedGroupLoss
from fairlearn.reductions import GridSearch
from fairlearn.reductions.grid_search.simple_quality_metrics import SimpleClassificationQualityMetric, SimpleRegressionQualityMetric  # noqa: E501

# ==============================================================

# The following are functions which convert ndarrays into other datatypes
# They are used to generate different argument types for calls to
# GridSearch


def identity(X):
    return X


def pandasdf(X):
    return pd.DataFrame(X)


def pandasseries(X):
    # Will not work if ndarray has more than one dimension
    return pd.Series(X)


def ndarray2d(X):
    # Adds a second dimension of length 1 onto a 1D
    # array. This is for checking that shapes n and
    # n*1 behave the same
    if len(X.shape) != 1:
        raise RuntimeError("ndarray2d requires 1d ndarray")

    X = np.expand_dims(X, 1)
    assert len(X.shape) == 2
    return X


# List the different datatypes which need to succeed for
# all GridSearch calls
candidate_X_transforms = [identity, pandasdf]
candidate_Y_transforms = [identity, pandasdf, pandasseries, ndarray2d]
candidate_A_transforms = [identity, pandasdf, pandasseries, ndarray2d]


# Base class for tests
# Tests which must be passed by all calls to the GridSearch
# go here
class ArgumentTests:
    def _quick_data(self):
        # Data are random and do not matter for these tests
        feature_1 = [0, 1, 2, 3, 4, 5, 6, 7]
        feature_2 = [5, 4, 3, 2, 7, 8, 3, 4]
        feature_3 = [9, 2, 4, 2, 9, 3, 1, 8]
        X = np.stack((feature_1, feature_2, feature_3), -1)
        Y = np.array([0, 1, 0, 1, 1, 1, 1, 0])
        A = np.array([1, 0, 0, 0, 0, 1, 1, 1])
        return X, Y, A

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_valid_inputs(self, transformX, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()
        gs.fit(transformX(X),
               transformY(Y),
               aux_data=transformA(A),
               number_of_lagrange_multipliers=2)
        assert len(gs.all_results) == 2

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    def test_X_is_None(self, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        _, Y, A = self._quick_data()

        message = str("Must supply X")
        with pytest.raises(ValueError) as execInfo:
            gs.fit(None,
                   transformY(Y),
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_Y_is_None(self, transformX, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, _, A = self._quick_data()

        message = str("Must supply y")
        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X),
                   None,
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_X_Y_different_rows(self, transformX, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, _, A = self._quick_data()
        Y = np.random.randint(2, size=len(A)+1)

        message = str("X and y must have same number of rows")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_X_A_different_rows(self, transformX, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, _ = self._quick_data()
        A = np.random.randint(2, size=len(Y)+1)

        message = str("X and the target attribute must have same number of rows")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_aux_data_non_binary(self, transformX, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()
        A[0] = 0
        A[1] = 1
        A[2] = 2

        message = str("Target Attribute contains more than two unique values")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_Y_df_bad_columns(self, transformX, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()

        Y_two_col_df = pd.DataFrame({"a": Y, "b": Y})
        message = str("y is a DataFrame with more than one column")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   Y_two_col_df,
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_Y_ndarray_bad_columns(self, transformX, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()

        Y_two_col_ndarray = np.stack((Y, Y), -1)
        message = str("y is an ndarray with more than one column")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   Y_two_col_ndarray,
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_A_df_bad_columns(self, transformX, transformY):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()

        A_two_col_df = pd.DataFrame({"a": A, "b": A})
        message = str("aux_data is a DataFrame with more than one column")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   aux_data=A_two_col_df,
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_A_ndarray_bad_columns(self, transformX, transformY):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()

        A_two_col_ndarray = np.stack((A, A), -1)
        message = str("aux_data is an ndarray with more than one column")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   aux_data=A_two_col_ndarray,
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]


# Tests specific to DemographicParity
class TestDemographicParity(ArgumentTests):
    def setup_method(self, method):
        logging.info("setup_method      method:%s" % method.__name__)
        self.learner = LogisticRegression()
        self.disparity_criterion = DemographicParity()
        self.quality_metric = SimpleClassificationQualityMetric()

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_Y_ternary(self, transformX, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()
        Y[0] = 0
        Y[1] = 1
        Y[2] = 2

        message = str("Supplied y labels are not 0 or 1")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_Y_not_0_1(self, transformX, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()
        Y = Y + 1

        message = str("Supplied y labels are not 0 or 1")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(transformX(X),
                   transformY(Y),
                   aux_data=transformA(A),
                   number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]


# Tests specific to BoundedGroupLoss
class TestBoundedGroupLoss(ArgumentTests):
    def setup_method(self, method):
        logging.info("setup_method      method:%s" % method.__name__)
        self.learner = LinearRegression()
        self.disparity_criterion = BoundedGroupLoss()
        self.quality_metric = SimpleRegressionQualityMetric()
