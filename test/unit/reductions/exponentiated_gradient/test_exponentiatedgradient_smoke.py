# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import (
    AbsoluteLoss,
    BoundedGroupLoss,
    DemographicParity,
    EqualizedOdds,
    ErrorRate,
    ErrorRateParity,
    ExponentiatedGradient,
    FalsePositiveRateParity,
    SquareLoss,
    TruePositiveRateParity,
    ZeroOneLoss,
)
from fairlearn.reductions._exponentiated_gradient._constants import _MIN_ITER
from fairlearn.reductions._moments.bounded_group_loss import MeanLoss

from .simple_learners import LeastSquaresBinaryClassifierLearner, LeastSquaresRegressor
from .test_utilities import _get_data

_PRECISION = 1e-5


class TestExponentiatedGradientSmoke:
    smoke_test_data = [
        {
            "constraint_class": DemographicParity,
            "eps": 0.100,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.100000,
            "error": 0.250000,
            "n_oracle_calls": 32,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.100,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": -0.020000,
            "error": 0.250000,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 2,
            "ratio": 0.8,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.050,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.266522,
            "n_oracle_calls": 23,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.050,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": -0.020000,
            "error": 0.25,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 2,
            "ratio": 0.8,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.020,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.332261,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.020,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": -0.020000,
            "error": 0.25,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 2,
            "ratio": 0.8,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.010,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.010000,
            "error": 0.354174,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.010,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": -0.020000,
            "error": 0.25,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 2,
            "ratio": 0.8,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.005,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.365130,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.005,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": -0.020000,
            "error": 0.25,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 2,
            "ratio": 0.8,
        },
        # ================================================
        {
            "constraint_class": DemographicParity,
            "eps": 0.050,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.407142,
            "n_oracle_calls": 18,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 4,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.050,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.263830,
            "n_oracle_calls": 21,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 3,
            "ratio": 0.8,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.020,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.422,
            "n_oracle_calls": 19,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
        },
        {
            "constraint_class": DemographicParity,
            "eps": 0.020,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.286170,
            "n_oracle_calls": 21,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 3,
            "ratio": 0.8,
        },
        # ================================================
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.100,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.100000,
            "error": 0.309333,
            "n_oracle_calls": 21,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 4,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.100,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.100000,
            "error": 0.25,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 4,
            "ratio": 0.8,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.050,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.378827,
            "n_oracle_calls": 19,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.050,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.277016,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 4,
            "ratio": 0.8,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.020,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.421531,
            "n_oracle_calls": 19,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.020,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.296612,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 4,
            "ratio": 0.8,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.010,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.010000,
            "error": 0.435765,
            "n_oracle_calls": 19,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.010,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.010000,
            "error": 0.303145,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 4,
            "ratio": 0.8,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.005,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.442883,
            "n_oracle_calls": 19,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.005,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.306411,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 4,
            "ratio": 0.8,
        },
        # ================================================
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.050,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.4125,
            "n_oracle_calls": 23,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.050,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.324067,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 4,
            "ratio": 0.8,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.020,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.435,
            "n_oracle_calls": 23,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.020,
            "objective": ErrorRate(costs={"fn": 2.0, "fp": 1.0}),
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.339179,
            "n_oracle_calls": 22,
            "n_oracle_calls_dummy_returned": 12,
            "n_predictors": 4,
            "ratio": 0.8,
        },
        # ================================================
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.1,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.100000,
            "error": 0.25625,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.1,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.092857,
            "error": 0.25,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
            "ratio": 0.8,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.05,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.049999,
            "error": 0.3,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.05,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.050000,
            "error": 0.253472,
            "n_oracle_calls": 26,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
            "ratio": 0.8,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.02,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.019999,
            "error": 0.326250,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.02,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.020000,
            "error": 0.268055,
            "n_oracle_calls": 26,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
            "ratio": 0.8,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.01,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.010000,
            "error": 0.325555,
            "n_oracle_calls": 18,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 4,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.01,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.010000,
            "error": 0.272916,
            "n_oracle_calls": 26,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
            "ratio": 0.8,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.005,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.329444,
            "n_oracle_calls": 19,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
        },
        {
            "constraint_class": ErrorRateParity,
            "eps": 0.005,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.275347,
            "n_oracle_calls": 26,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 5,
            "ratio": 0.8,
        },
        # ================================================
        {
            "constraint_class": TruePositiveRateParity,
            "eps": 0.005,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.0,
            "error": 0.25,
            "n_oracle_calls": 16,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 2,
        },
        {
            "constraint_class": FalsePositiveRateParity,
            "eps": 0.005,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.427133,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
    ]
    smoke_test_data_flipped = [
        {
            "constraint_class": TruePositiveRateParity,
            "eps": 0.005,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.427133,
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
        {
            "constraint_class": FalsePositiveRateParity,
            "eps": 0.005,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.0,
            "error": 0.25,
            "n_oracle_calls": 16,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 2,
        },
        {
            "constraint_class": EqualizedOdds,
            "eps": 0.005,
            "best_gap": 0.000000,
            "last_iter": 5,
            "best_iter": 5,
            "disp": 0.005000,
            "error": 0.442883,
            "n_oracle_calls": 19,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 6,
        },
    ]

    smoke_test_data_regression = [
        {
            "constraint_class": BoundedGroupLoss,
            "loss": SquareLoss(0, 1),
            "eps": 0.01,
            "best_gap": 0.003905,
            "last_iter": 6,
            "best_iter": 6,
            "upper_bound": 0.01,  # infeasible
            "disp": [
                0.178333,
                0.178333,
                0.178333,
                0.178333,
                0.178333,
                0.178333,
                0.028045,
                0.178333,
                0.178333,
                0.178333,
                0.030853,
                0.178333,
                0.178333,
                0.178333,
                0.178333,
                0.178333,
            ],
            "error": [
                0.1035,
                0.1035,
                0.1035,
                0.1035,
                0.1035,
                0.1035,
                0.024412,
                0.1035,
                0.1035,
                0.1035,
                0.025691,
                0.1035,
                0.1035,
                0.1035,
                0.1035,
                0.1035,
            ],
            "weights": [
                0,
                0,
                0,
                0,
                0,
                0,
                0.956748,
                0,
                0,
                0,
                0.043251,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            "n_oracle_calls": 23,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 16,
        },
        {
            "constraint_class": BoundedGroupLoss,
            "loss": SquareLoss(0, 1),
            "eps": 0.01,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "upper_bound": 0.05,  # feasible
            "disp": [
                0.178333,
                0.178333,
                0.036690,
                0.178333,
                0.178333,
                0.178333,
                0.178333,
            ],
            "error": [0.1035, 0.1035, 0.021988, 0.1035, 0.1035, 0.1035, 0.1035],
            "weights": [0, 0, 1, 0, 0, 0, 0],
            "n_oracle_calls": 32,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 7,
        },
        {
            "constraint_class": BoundedGroupLoss,
            "loss": SquareLoss(0, 1),
            "eps": 0.01,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "max_iter": 20,
            "nu": 1e-6,
            "upper_bound": 0.05,  # feasible
            "disp": [
                0.178333,
                0.178333,
                0.036690,
                0.178333,
                0.178333,
                0.178333,
                0.178333,
            ],
            "error": [0.1035, 0.1035, 0.021988, 0.1035, 0.1035, 0.1035, 0.1035],
            "weights": [0, 0, 1, 0, 0, 0, 0],
            "n_oracle_calls": 29,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 7,
        },
        {
            "constraint_class": BoundedGroupLoss,
            "loss": ZeroOneLoss(),
            "eps": 0.01,
            "best_gap": 0.007185,
            "last_iter": 5,
            "best_iter": 5,
            "upper_bound": 0.01,  # infeasible
            "disp": [
                0.383333,
                0.383333,
                0.383333,
                0.383333,
                0.1479,
                0.383333,
                0.383333,
                0.383333,
                0.140256,
                0.383333,
                0.383333,
                0.383333,
                0.383333,
                0.383333,
            ],
            "error": [
                0.255,
                0.255,
                0.255,
                0.255,
                0.140198,
                0.255,
                0.255,
                0.255,
                0.135674,
                0.255,
                0.255,
                0.255,
                0.255,
                0.255,
            ],
            "weights": [0, 0, 0, 0, 0.221468, 0, 0, 0, 0.778531, 0, 0, 0, 0, 0],
            "n_oracle_calls": 20,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 14,
        },
        {
            "constraint_class": BoundedGroupLoss,
            "loss": ZeroOneLoss(),
            "eps": 0.01,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "upper_bound": 0.2,  # feasible
            "disp": [0.383333, 0.383333, 0.166918],
            "error": [0.255, 0.255, 0.116949],
            "weights": [0, 0, 1],
            "n_oracle_calls": 20,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
        {
            "constraint_class": BoundedGroupLoss,
            "loss": ZeroOneLoss(),
            "eps": 0.01,
            "best_gap": 0.0,
            "last_iter": 5,
            "best_iter": 5,
            "max_iter": 20,
            "nu": 1e-6,
            "upper_bound": 0.2,  # feasible
            "disp": [0.383333, 0.383333, 0.166918],
            "error": [0.255, 0.255, 0.116949],
            "weights": [0, 0, 1],
            "n_oracle_calls": 17,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 3,
        },
        {
            "constraint_class": BoundedGroupLoss,
            "loss": AbsoluteLoss(0, 1),
            "eps": 0.01,
            "best_gap": 0.007185,
            "last_iter": 5,
            "best_iter": 5,
            "upper_bound": 0.01,  # infeasible
            "disp": [
                0.383333,
                0.383333,
                0.383333,
                0.383333,
                0.1479,
                0.383333,
                0.383333,
                0.383333,
                0.140256,
                0.383333,
                0.383333,
                0.383333,
                0.383333,
                0.383333,
            ],
            "error": [
                0.255,
                0.255,
                0.255,
                0.255,
                0.140198,
                0.255,
                0.255,
                0.255,
                0.135674,
                0.255,
                0.255,
                0.255,
                0.255,
                0.255,
            ],
            "weights": [0, 0, 0, 0, 0.221468, 0, 0, 0, 0.778531, 0, 0, 0, 0, 0],
            "n_oracle_calls": 20,
            "n_oracle_calls_dummy_returned": 0,
            "n_predictors": 14,
        },
    ]

    def run_smoke_test_binary_classification(self, data, flipped=False):
        learner = LeastSquaresBinaryClassifierLearner()
        if "ratio" in data.keys():
            disparity_moment = data["constraint_class"](
                ratio_bound_slack=data["eps"], ratio_bound=data["ratio"]
            )
        else:
            disparity_moment = data["constraint_class"](difference_bound=data["eps"])
        if "objective" in data.keys():
            objective_moment = deepcopy(data["objective"])
        else:
            objective_moment = ErrorRate()

        # Create Exponentiated Gradient object with a copy of the constraint.
        # The original disparity_moment object is used for validation, so the
        # assumption is that the moment logic is correct in these tests.
        expgrad = ExponentiatedGradient(
            learner,
            constraints=deepcopy(disparity_moment),
            objective=deepcopy(objective_moment),
            eps=data["eps"],
        )

        X, y, A = _get_data(A_two_dim=False, flip_y=flipped)

        expgrad.fit(X, y, sensitive_features=A)

        self._assert_expgrad_state(expgrad, data)

        # select probability of predicting 1
        def Q(X):
            return expgrad._pmf_predict(X)[:, 1]

        disparity_moment.load_data(X, y, sensitive_features=A)
        objective_moment.load_data(X, y, sensitive_features=A)
        disparity = disparity_moment.gamma(Q).max()
        error = objective_moment.gamma(Q).iloc[0]
        assert disparity == pytest.approx(data["disp"], abs=_PRECISION)
        assert error == pytest.approx(data["error"], abs=_PRECISION)

    @pytest.mark.parametrize("testdata", smoke_test_data)
    def test_smoke(self, testdata):
        self.run_smoke_test_binary_classification(testdata)

    @pytest.mark.parametrize("testdata", smoke_test_data_flipped)
    def test_smoke_flipped(self, testdata):
        self.run_smoke_test_binary_classification(testdata, flipped=True)

    @pytest.mark.parametrize("data", smoke_test_data_regression)
    def test_smoke_regression(self, data):
        learner = LeastSquaresRegressor()
        disparity_moment = data["constraint_class"](
            loss=data["loss"], upper_bound=data["upper_bound"]
        )

        # Create Exponentiated Gradient object with a copy of the constraint.
        # The original disparity_moment object is used for validation, so the
        # assumption is that the moment logic is correct in these tests.
        expgrad = ExponentiatedGradient(
            learner,
            constraints=deepcopy(disparity_moment),
            eps=data["eps"],
            nu=data.get("nu"),
            max_iter=data.get("max_iter", 50),
        )

        X, y, A = _get_data(A_two_dim=False, y_as_scores=True)

        expgrad.fit(X, y, sensitive_features=A)

        self._assert_expgrad_state(expgrad, data)

        # check all predictors
        disparity_moment.load_data(X, y, sensitive_features=A)
        for i in range(len(expgrad.predictors_)):

            def Q(X):
                return expgrad._pmf_predict(X)[i]

            default_objective = MeanLoss(data["loss"])
            default_objective.load_data(X, y, sensitive_features=A)
            disparity = disparity_moment.gamma(Q).max()
            error = default_objective.gamma(Q).iloc[0]
            assert disparity == pytest.approx(data["disp"][i], abs=_PRECISION)
            assert error == pytest.approx(data["error"][i], abs=_PRECISION)
            assert expgrad.weights_[i] == pytest.approx(
                data["weights"][i], abs=_PRECISION
            )

        assert sum(expgrad.weights_) == pytest.approx(1, abs=_PRECISION)

    @pytest.mark.parametrize(
        "Constraints",
        [
            TruePositiveRateParity,
            FalsePositiveRateParity,
            DemographicParity,
            EqualizedOdds,
            ErrorRateParity,
        ],
    )
    def test_simple_fit_predict_binary_classification(self, Constraints):
        X, y, sensitive_features = _get_data()
        estimator = LeastSquaresBinaryClassifierLearner()
        expgrad = ExponentiatedGradient(estimator, Constraints())
        expgrad.fit(X, y, sensitive_features=sensitive_features)
        expgrad.predict(X)

    @pytest.mark.parametrize(
        "constraints",
        [
            BoundedGroupLoss(loss=SquareLoss(0, 1), upper_bound=0.01),
            BoundedGroupLoss(loss=AbsoluteLoss(0, 1), upper_bound=0.01),
            BoundedGroupLoss(loss=ZeroOneLoss(), upper_bound=0.01),
        ],
    )
    def test_simple_fit_predict_regression(self, constraints):
        X, y, sensitive_features = _get_data(y_as_scores=True)
        estimator = LeastSquaresRegressor()
        expgrad = ExponentiatedGradient(estimator, constraints)
        expgrad.fit(X, y, sensitive_features=sensitive_features)
        expgrad.predict(X)

    def test_single_y_value(self):
        # Setup with data designed to result in "all single class"
        # at some point in the grid
        X_dict = {"c": [10, 50, 10]}
        X = pd.DataFrame(X_dict)

        y = [1, 1, 1]
        A = ["a", "b", "b"]

        estimator = LogisticRegression(
            solver="liblinear", fit_intercept=True, random_state=97
        )
        expgrad = ExponentiatedGradient(estimator, DemographicParity())

        # Following line should not throw an exception
        expgrad.fit(X, y, sensitive_features=A)

        # Check the predictors for a ConstantPredictor
        test_X_dict = {"c": [1, 2, 3, 4, 5, 6]}
        test_X = pd.DataFrame(test_X_dict)
        assert expgrad.n_oracle_calls_dummy_returned_ > 0
        assert len(expgrad.oracle_execution_times_) == expgrad.n_oracle_calls_
        for p in expgrad.predictors_:
            assert isinstance(p, DummyClassifier)
            assert np.array_equal(p.predict(test_X), [1, 1, 1, 1, 1, 1])

    def _assert_expgrad_state(self, expgrad, data):
        n_predictors = len(expgrad.predictors_)
        assert expgrad.best_gap_ == pytest.approx(data["best_gap"], abs=_PRECISION)
        assert expgrad.best_gap_ < expgrad.nu
        assert expgrad.last_iter_ == data["last_iter"]
        assert expgrad.best_iter_ == data["best_iter"]
        assert expgrad.last_iter_ >= _MIN_ITER
        assert expgrad.n_oracle_calls_ == data["n_oracle_calls"]
        assert (
            expgrad.n_oracle_calls_dummy_returned_
            == data["n_oracle_calls_dummy_returned"]
        )
        assert n_predictors == data["n_predictors"]
        assert len(expgrad.oracle_execution_times_) == expgrad.n_oracle_calls_

    @pytest.mark.parametrize("eps", [0.05, 0.02])
    @pytest.mark.parametrize("ratio", [None, 0.8])
    @pytest.mark.parametrize("pos_copies", [0, 1, 2])
    def test_error_rate_consistency(self, eps, ratio, pos_copies):
        learner = LeastSquaresBinaryClassifierLearner()
        if ratio is None:
            constraints_moment = EqualizedOdds(difference_bound=eps)
        else:
            constraints_moment = EqualizedOdds(ratio_bound=ratio, ratio_bound_slack=eps)

        results = {}
        for method in ["costs", "sampling"]:
            X, y, A = _get_data()

            if method == "sampling":
                select = y == 1
                X = pd.concat((X,) + (X.loc[select, :],) * pos_copies).values
                y = pd.concat((y,) + (y[select],) * pos_copies).values
                A = pd.concat((A,) + (A[select],) * pos_copies).values
                objective_moment = ErrorRate()
            else:
                objective_moment = ErrorRate(costs={"fn": 1.0 + pos_copies, "fp": 1.0})

            expgrad = ExponentiatedGradient(
                learner,
                constraints=deepcopy(constraints_moment),
                objective=deepcopy(objective_moment),
                eps=eps,
                nu=1e-3,
            )

            expgrad.fit(X, y, sensitive_features=A)

            # select probability of predicting 1
            def Q(X):
                return expgrad._pmf_predict(X)[:, 1]

            constraints_eval = deepcopy(constraints_moment)
            constraints_eval.load_data(X, y, sensitive_features=A)
            disparity = constraints_eval.gamma(Q).max()

            objective_eval = deepcopy(objective_moment)
            objective_eval.load_data(X, y, sensitive_features=A)
            total_error = objective_eval.gamma(Q).iloc[0] * len(y)
            results[method] = {
                "error": objective_eval.gamma(Q).iloc[0],
                "total_error": total_error,
                "disp": disparity,
                "n_predictors": len(expgrad.predictors_),
                "best_gap": expgrad.best_gap_,
                "last_iter": expgrad.last_iter_,
                "best_iter": expgrad.best_iter_,
                "n_oracle_calls": expgrad.n_oracle_calls_,
                "n_oracle_calls_dummy_returned": expgrad.n_oracle_calls_dummy_returned_,
            }

        self._assert_expgrad_two_states(results["costs"], results["sampling"])

    def _assert_expgrad_two_states(self, state1, state2):
        assert state1["total_error"] == pytest.approx(
            state2["total_error"], abs=_PRECISION
        )
        assert state1["disp"] == pytest.approx(state2["disp"], abs=_PRECISION)
        assert state1["n_predictors"] == state2["n_predictors"]
        assert state1["best_gap"] == pytest.approx(state2["best_gap"], abs=_PRECISION)
        assert state1["last_iter"] == state2["last_iter"]
        assert state1["best_iter"] == state2["best_iter"]
        assert state1["n_oracle_calls"] == state2["n_oracle_calls"]
        assert (
            state1["n_oracle_calls_dummy_returned"]
            == state2["n_oracle_calls_dummy_returned"]
        )
