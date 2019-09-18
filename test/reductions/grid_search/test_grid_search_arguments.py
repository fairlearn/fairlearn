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


def identity(X):
    return X


def pandasdf(X):
    return pd.DataFrame(X)


def pandasseries(X):
    return pd.Series(X)


def ndarray2d(X):
    if len(X.shape) != 1:
        raise RuntimeError("ndarray2d requires 1d ndarray")

    X.shape = (X.shape[0], 1)
    return X


class ArgumentTests:
    def _quick_data(self, number_samples=64):
        feature_1 = np.random.randint(2, size=number_samples)
        feature_2 = np.random.randint(6, size=number_samples)
        feature_3 = np.random.randint(3, size=number_samples)
        X = np.stack((feature_1, feature_2, feature_3), -1)
        Y = np.random.randint(2, size=number_samples)
        A = np.random.randint(2, size=number_samples)
        return X, Y, A

    Xtransform = [identity, pandasdf]
    Ytransform = [identity, pandasdf, pandasseries, ndarray2d]
    Atransform = [identity, pandasdf, pandasseries, ndarray2d]

    def run_smoke(self, X, Y, A):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)

        gs.fit(X, Y, aux_data=A, number_of_lagrange_multipliers=5)

        assert len(gs.all_results) == 5

    @pytest.mark.parametrize("transformA", Atransform)
    @pytest.mark.parametrize("transformY", Ytransform)
    @pytest.mark.parametrize("transformX", Xtransform)
    def test_valid_inputs(self, transformX, transformY, transformA):
        X, Y, A = self._quick_data()
        self.run_smoke(transformX(X), transformY(Y), transformA(A))

    @pytest.mark.parametrize("transformA", Atransform)
    @pytest.mark.parametrize("transformY", Ytransform)
    @pytest.mark.parametrize("transformX", Xtransform)
    def test_aux_data_non_binary(self, transformX, transformY, transformA):
        gs = GridSearch(self.learner, self.disparity_criterion, self.quality_metric)
        X, Y, A = self._quick_data()
        A[0] = 0
        A[1] = 1
        A[2] = 2

        message = str("Protected Attribute contains more than two unique values")
        with pytest.raises(RuntimeError) as execInfo:
            gs.fit(X, Y, aux_data=A, number_of_lagrange_multipliers=3)

        assert message == execInfo.value.args[0]


class TestDemographicParity(ArgumentTests):
    def setup_method(self, method):
        logging.info("setup_method      method:%s" % method.__name__)
        self.learner = LogisticRegression()
        self.disparity_criterion = DemographicParity()
        self.quality_metric = SimpleClassificationQualityMetric()


class TestBoundedGroupLoss(ArgumentTests):
    def setup_method(self, method):
        logging.info("setup_method      method:%s" % method.__name__)
        self.learner = LinearRegression()
        self.disparity_criterion = BoundedGroupLoss()
        self.quality_metric = SimpleRegressionQualityMetric()
