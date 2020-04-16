# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import pytest


from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions._exponentiated_gradient._constants import _MIN_T
from fairlearn.reductions import DemographicParity, EqualizedOdds, ErrorRateRatio,\
    TruePositiveRateDifference, ErrorRate
from .simple_learners import LeastSquaresBinaryClassifierLearner
from .test_utilities import sensitive_features, X1, labels, _get_data


class TestExponentiatedGradientSmoke:
    def setup_method(self, method):
        self.X, self.y, self.A = _get_data(A_two_dim=False)
        self.flipped_y = pd.Series([int(not i) for i in labels])
        self.learner = LeastSquaresBinaryClassifierLearner()
        self._PRECISION = 1e-6

    smoke_test_data = [{"cons_class": DemographicParity, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.250000, "n_oracle_calls": 32,
                        "n_predictors": 3},
                       {"cons_class": DemographicParity, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": -0.020000,
                        "error": 0.250000, "n_oracle_calls": 17,
                        "n_predictors": 2, "ratio": 0.8},
                       {"cons_class": DemographicParity, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.266522, "n_oracle_calls": 23,
                        "n_predictors": 6},
                       {"cons_class": DemographicParity, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": -0.020000,
                        "error": 0.25, "n_oracle_calls": 17,
                        "n_predictors": 2, "ratio": 0.8},
                       {"cons_class": DemographicParity, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.332261, "n_oracle_calls": 22,
                        "n_predictors": 5},
                       {"cons_class": DemographicParity, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": -0.020000,
                        "error": 0.25, "n_oracle_calls": 17,
                        "n_predictors": 2, "ratio": 0.8},
                       {"cons_class": DemographicParity, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.354174, "n_oracle_calls": 22,
                        "n_predictors": 5},
                       {"cons_class": DemographicParity, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": -0.020000,
                        "error": 0.25, "n_oracle_calls": 17,
                        "n_predictors": 2, "ratio": 0.8},
                       {"cons_class": DemographicParity, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.365130, "n_oracle_calls": 22,
                        "n_predictors": 5},
                       {"cons_class": DemographicParity, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": -0.020000,
                        "error": 0.25, "n_oracle_calls": 17,
                        "n_predictors": 2, "ratio": 0.8},
                       {"cons_class": EqualizedOdds, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.309333, "n_oracle_calls": 21,
                        "n_predictors": 4},
                       {"cons_class": EqualizedOdds, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.25, "n_oracle_calls": 22,
                        "n_predictors": 4, "ratio": 0.8},
                       {"cons_class": EqualizedOdds, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.378827, "n_oracle_calls": 19,
                        "n_predictors": 6},
                       {"cons_class": EqualizedOdds, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.277016, "n_oracle_calls": 22,
                        "n_predictors": 4, "ratio": 0.8},
                       {"cons_class": EqualizedOdds, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.421531, "n_oracle_calls": 19,
                        "n_predictors": 6},
                       {"cons_class": EqualizedOdds, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.296612, "n_oracle_calls": 22,
                        "n_predictors": 4, "ratio": 0.8},
                       {"cons_class": EqualizedOdds, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.435765, "n_oracle_calls": 19,
                        "n_predictors": 6},
                       {"cons_class": EqualizedOdds, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.303145, "n_oracle_calls": 22,
                        "n_predictors": 4, "ratio": 0.8},
                       {"cons_class": EqualizedOdds, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.442883, "n_oracle_calls": 19,
                        "n_predictors": 6},
                       {"cons_class": EqualizedOdds, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.306411, "n_oracle_calls": 22,
                        "n_predictors": 4, "ratio": 0.8},
                       {"cons_class": ErrorRateRatio, "eps": 0.1,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.25625, "n_oracle_calls": 17,
                        "n_predictors": 3},
                       {"cons_class": ErrorRateRatio, "eps": 0.1,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.092857,
                        "error": 0.25, "n_oracle_calls": 17,
                        "n_predictors": 3, "ratio": 0.8},
                       {"cons_class": ErrorRateRatio, "eps": 0.05,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.049999,
                        "error": 0.3, "n_oracle_calls": 17,
                        "n_predictors": 3},
                       {"cons_class": ErrorRateRatio, "eps": 0.05,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.253472, "n_oracle_calls": 26,
                        "n_predictors": 6, "ratio": 0.8},
                       {"cons_class": ErrorRateRatio, "eps": 0.02,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.019999,
                        "error": 0.326250, "n_oracle_calls": 17,
                        "n_predictors": 3},
                       {"cons_class": ErrorRateRatio, "eps": 0.02,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.268055, "n_oracle_calls": 26,
                        "n_predictors": 5, "ratio": 0.8},
                       {"cons_class": ErrorRateRatio, "eps": 0.01,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.325555, "n_oracle_calls": 18,
                        "n_predictors": 4},
                       {"cons_class": ErrorRateRatio, "eps": 0.01,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.272916, "n_oracle_calls": 26,
                        "n_predictors": 5, "ratio": 0.8},
                       {"cons_class": ErrorRateRatio, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.329444, "n_oracle_calls": 19,
                        "n_predictors": 5},
                       {"cons_class": ErrorRateRatio, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.275347, "n_oracle_calls": 26,
                        "n_predictors": 5, "ratio": 0.8},
                       {"cons_class": TruePositiveRateDifference, "eps": 0.005,
                        "best_gap": 0.0, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.25, "n_oracle_calls": 16,
                        "n_predictors": 2}
                       ]
    smoke_test_data_flipped = [
        {"cons_class": TruePositiveRateDifference, "eps": 0.005,
         "best_gap": 0.0, "last_t": 5,
         "best_t": 5, "disp": 0.005000,
         "error": 0.427133, "n_oracle_calls": 17,
         "n_predictors": 3},
        {"cons_class": EqualizedOdds, "eps": 0.005,
         "best_gap": 0.000000, "last_t": 5,
         "best_t": 5, "disp": 0.005000,
         "error": 0.442883, "n_oracle_calls": 19,
         "n_predictors": 6}
    ]

    def run_smoke_test(self, data, flipped=False):
        ratio = 1.0
        if "ratio" in data.keys():
            ratio = data["ratio"]
        expgrad = ExponentiatedGradient(self.learner, constraints=data["cons_class"](ratio=ratio),
                                        eps=data["eps"])
        expgrad.fit(self.X, (self.flipped_y if flipped else self.y), sensitive_features=self.A)

        def Q(X): return expgrad._pmf_predict(X)[:, 1]
        n_predictors = len(expgrad._predictors)

        disparity_moment = data["cons_class"](ratio=ratio)
        disparity_moment.load_data(self.X, (self.flipped_y if flipped else self.y),
                                   sensitive_features=self.A)
        error = ErrorRate()
        error.load_data(self.X, (self.flipped_y if flipped else self.y), sensitive_features=self.A)
        disparity = disparity_moment.gamma(Q).max()
        error = error.gamma(Q)[0]

        assert expgrad._best_gap == pytest.approx(data["best_gap"], abs=self._PRECISION)
        assert expgrad._last_t == data["last_t"]
        assert expgrad._best_t == data["best_t"]
        assert expgrad._last_t >= _MIN_T
        assert disparity == pytest.approx(data["disp"], abs=self._PRECISION)
        assert error == pytest.approx(data["error"], abs=self._PRECISION)
        assert expgrad._n_oracle_calls == data["n_oracle_calls"]
        assert n_predictors == data["n_predictors"]

    @pytest.mark.parametrize("testdata", smoke_test_data)
    def test_smoke(self, testdata):
        self.run_smoke_test(testdata)

    @pytest.mark.parametrize("testdata", smoke_test_data_flipped)
    def test_smoke_flipped(self, testdata):
        self.run_smoke_test(testdata, flipped=True)

    @pytest.mark.parametrize("constraint", [TruePositiveRateDifference, DemographicParity])
    def test_simple_fit_predict(self, constraint):
        estimator = LeastSquaresBinaryClassifierLearner()
        expgrad = ExponentiatedGradient(estimator, constraint())
        expgrad.fit(pd.DataFrame(X1), pd.Series(labels),
                    sensitive_features=pd.Series(sensitive_features))
        expgrad.predict(pd.DataFrame(X1))
