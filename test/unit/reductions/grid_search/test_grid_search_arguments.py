# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging
from test.unit.fixes import get_sklearn_expected_1d_message
from test.unit.input_convertors import (
    conversions_for_1d,
    ensure_dataframe,
    ensure_ndarray,
)
from test.unit.reductions.conftest import is_invalid_transformation
from test.unit.reductions.grid_search.utilities import (
    _quick_data,
    assert_n_grid_search_results,
)

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fairlearn.reductions import (
    BoundedGroupLoss,
    DemographicParity,
    EqualizedOdds,
    GridSearch,
    ZeroOneLoss,
)
from fairlearn.reductions._grid_search._grid_generator import (
    GRID_DIMENSION_WARN_TEMPLATE,
    GRID_DIMENSION_WARN_THRESHOLD,
    GRID_SIZE_WARN_TEMPLATE,
    _GridGenerator,
)
from fairlearn.utils._input_validation import (
    _LABELS_NOT_0_1_ERROR_MESSAGE,
    _MESSAGE_Y_NONE,
)

# ==============================================================

# List the different datatypes which need to succeed for
# all GridSearch calls
candidate_X_transforms = [ensure_ndarray, ensure_dataframe]
candidate_Y_transforms = conversions_for_1d
candidate_A_transforms = conversions_for_1d

# ==============================================================

not_fitted_error_msg = (
    "This {} instance is not fitted yet. Call 'fit' with "
    "appropriate arguments before using this estimator."
)


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
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            grid_size=2,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, A = _quick_data(A_two_dim)
        gs.fit(transformX(X), transformY(Y), sensitive_features=transformA(A))
        assert_n_grid_search_results(2, gs)

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_X_is_None(self, transformY, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            grid_size=3,
            sample_weight_name=self.sample_weight_name,
        )
        _, Y, A = _quick_data(A_two_dim)

        with pytest.raises(ValueError) as execInfo:
            gs.fit(None, transformY(Y), sensitive_features=transformA(A))

        assert "Expected 2D array, got scalar array instead" in execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_is_None(self, transformX, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, _, A = _quick_data()

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X), None, sensitive_features=transformA(A))

        assert _MESSAGE_Y_NONE == execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_X_Y_different_rows(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, _, A = _quick_data()
        Y = np.random.randint(2, size=len(A) + 1)

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X), transformY(Y), sensitive_features=transformA(A))

        expected_exception_message = (
            "Found input variables with inconsistent numbers of samples"
        )
        assert expected_exception_message in execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_X_A_different_rows(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, _ = _quick_data(A_two_dim)
        A = np.random.randint(2, size=len(Y) + 1)
        if A_two_dim:
            A = np.stack((A, A), -1)

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X), transformY(Y), sensitive_features=transformA(A))

        expected_exception_message = (
            "Found input variables with inconsistent numbers of samples"
        )
        assert expected_exception_message in execInfo.value.args[0]

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_many_sensitive_feature_groups_warning(
        self, transformX, transformY, transformA, A_two_dim, caplog
    ):
        # The purpose of this test case is to create enough groups to trigger certain expected
        # warnings. The scenario should still work and succeed.
        grid_size = 10
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            grid_size=grid_size,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, A = _quick_data(A_two_dim)

        if A_two_dim:
            A[0][0] = 0
            A[0][1] = 0
            A[1][0] = 1
            A[1][1] = 1
            A[2][0] = 2
            A[2][1] = 2
            A[3][0] = 3
            A[3][1] = 3
            A[4][0] = 4
            A[4][1] = 4
            A[5][0] = 5
            A[5][1] = 5
        else:
            A[0] = 0
            A[1] = 1
            A[2] = 2
            A[3] = 3
            A[4] = 4
            A[5] = 5

        caplog.set_level(logging.WARNING)
        gs.fit(transformX(X), transformY(Y), sensitive_features=transformA(A))

        log_records = caplog.get_records("call")
        dimension_log_record = log_records[0]
        size_log_record = log_records[1]
        if isinstance(self.disparity_criterion, EqualizedOdds):
            # not every label occurs with every group
            grid_dimensions = 10
        else:
            # 6 groups total, but one is not part of the basis, so 5 dimensions
            grid_dimensions = 5

        # expect both the dimension warning and the grid size warning
        assert len(log_records) == 2
        assert GRID_DIMENSION_WARN_TEMPLATE.format(
            grid_dimensions, GRID_DIMENSION_WARN_THRESHOLD
        ) in dimension_log_record.msg.format(*dimension_log_record.args)
        assert GRID_SIZE_WARN_TEMPLATE.format(
            grid_size, 2**grid_dimensions
        ) in size_log_record.msg.format(*size_log_record.args)

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.parametrize("n_groups", [2, 3, 4, 5])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_grid_size_warning_up_to_5_sensitive_feature_group(
        self, transformX, transformY, transformA, A_two_dim, n_groups, caplog
    ):
        if isinstance(self.disparity_criterion, EqualizedOdds):
            pytest.skip(
                "With EqualizedOdds there would be multiple warnings due to higher grid"
                " dimensionality."
            )

        grid_size = 10
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            grid_size=grid_size,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, A = _quick_data(A_two_dim, n_groups=n_groups)

        caplog.set_level(logging.WARNING)
        gs.fit(transformX(X), transformY(Y), sensitive_features=transformA(A))

        # don't expect the dimension warning;
        # but expect the grid size warning for large numbers of groups
        log_records = caplog.get_records("call")

        # 6 groups total, but one is not part of the basis, so 5 dimensions
        grid_dimensions = n_groups - 1

        if 2 ** (n_groups - 1) > grid_size:
            assert len(log_records) == 1
            size_log_record = log_records[0]
            assert GRID_SIZE_WARN_TEMPLATE.format(
                grid_size, 2**grid_dimensions
            ) in size_log_record.msg.format(*size_log_record.args)
        else:
            assert len(log_records) == 0

    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    def test_custom_grid(self, transformX, transformY, transformA):
        # Creating a standard grid with the default parameters
        grid_size = 10
        grid_limit = 2.0
        grid_offset = 0.1

        disparity_moment = EqualizedOdds()
        X, y, A = _quick_data(False)

        disparity_moment.load_data(X, y, sensitive_features=A)

        grid = _GridGenerator(
            grid_size,
            grid_limit,
            disparity_moment.pos_basis,
            disparity_moment.neg_basis,
            disparity_moment.neg_basis_present,
            False,
            grid_offset,
        ).grid

        # Creating a custom grid by selecting only a few columns from the grid to try out
        indices = [7, 3, 4]
        grid = grid.iloc[:, indices]

        gs = GridSearch(
            estimator=LogisticRegression(solver="liblinear"),
            constraints=EqualizedOdds(),
            grid=grid,
        )

        # Check that fit runs successfully with the custom grid
        gs.fit(transformX(X), transformY(y), sensitive_features=transformA(A))

        # Check that it trained the correct number of predictors
        assert len(gs.predictors_) == len(grid.columns)

    # ----------------------------

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_df_bad_columns(self, transformX, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, A = _quick_data(A_two_dim)

        Y_two_col_df = pd.DataFrame({"a": Y, "b": Y})
        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X), Y_two_col_df, sensitive_features=transformA(A))
        assert get_sklearn_expected_1d_message() in execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_ndarray_bad_columns(self, transformX, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, A = _quick_data(A_two_dim)

        Y_two_col_ndarray = np.stack((Y, Y), -1)
        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X), Y_two_col_ndarray, sensitive_features=transformA(A))
        assert get_sklearn_expected_1d_message() in execInfo.value.args[0]

    # ----------------------------

    def test_no_predict_before_fit(self):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, _, _ = _quick_data()

        with pytest.raises(NotFittedError) as execInfo:
            gs.predict(X)

        assert (
            not_fitted_error_msg.format(GridSearch.__name__) == execInfo.value.args[0]
        )

    def test_no_predict_proba_before_fit(self):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, _, _ = _quick_data()

        with pytest.raises(NotFittedError) as execInfo:
            gs.predict_proba(X)

        assert (
            not_fitted_error_msg.format(GridSearch.__name__) == execInfo.value.args[0]
        )


# Tests specific to Classification
class ConditionalOpportunityTests(ArgumentTests):
    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_ternary(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, A = _quick_data(A_two_dim)
        Y[0] = 0
        Y[1] = 1
        Y[2] = 2

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X), transformY(Y), sensitive_features=transformA(A))

        assert _LABELS_NOT_0_1_ERROR_MESSAGE == execInfo.value.args[0]

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    @pytest.mark.parametrize("A_two_dim", [False, True])
    @pytest.mark.uncollect_if(func=is_invalid_transformation)
    def test_Y_not_0_1(self, transformX, transformY, transformA, A_two_dim):
        gs = GridSearch(
            self.estimator,
            self.disparity_criterion,
            sample_weight_name=self.sample_weight_name,
        )
        X, Y, A = _quick_data(A_two_dim)
        Y = Y + 1

        with pytest.raises(ValueError) as execInfo:
            gs.fit(transformX(X), transformY(Y), sensitive_features=transformA(A))

        assert _LABELS_NOT_0_1_ERROR_MESSAGE == execInfo.value.args[0]


# Set up Pipeline estimator
class TestPipelineEstimator(ConditionalOpportunityTests):
    def setup_method(self, method):
        self.estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(solver="liblinear")),
            ]
        )
        self.disparity_criterion = DemographicParity()
        self.sample_weight_name = "logistic__sample_weight"


# Set up DemographicParity
class TestDemographicParity(ConditionalOpportunityTests):
    def setup_method(self, method):
        self.estimator = LogisticRegression(solver="liblinear")
        self.disparity_criterion = DemographicParity()
        self.sample_weight_name = "sample_weight"


# Test EqualizedOdds
class TestEqualizedOdds(ConditionalOpportunityTests):
    def setup_method(self, method):
        self.estimator = LogisticRegression(solver="liblinear")
        self.disparity_criterion = EqualizedOdds()
        self.sample_weight_name = "sample_weight"


# Tests specific to BoundedGroupLoss
class TestBoundedGroupLoss(ArgumentTests):
    def setup_method(self, method):
        self.estimator = LinearRegression()
        eps = 0.01
        self.disparity_criterion = BoundedGroupLoss(ZeroOneLoss(), upper_bound=eps)
        self.sample_weight_name = "sample_weight"
