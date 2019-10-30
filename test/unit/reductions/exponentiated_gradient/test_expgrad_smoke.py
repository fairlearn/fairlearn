# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import pytest


from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity, EqualizedOdds
from fairlearn.reductions import ErrorRate
from simple_learners import LeastSquaresBinaryClassifierLearner
from test_utilities import sensitive_features, X1, X2, X3, labels


class TestExpgradSmoke:
    def setup_method(self, method):
        print("setup_method      method:%s" % method.__name__)
        self.X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
        self.y = pd.Series(labels)
        self.A = pd.Series(sensitive_features)
        self.learner = LeastSquaresBinaryClassifierLearner()
        self._PRECISION = 1e-6

    smoke_test_data = [{"cons_class": DemographicParity, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.250000, "n_oracle_calls": 32,
                        "n_classifiers": 3},
                       {"cons_class": DemographicParity, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.266522, "n_oracle_calls": 23,
                        "n_classifiers": 6},
                       {"cons_class": DemographicParity, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.332261, "n_oracle_calls": 22,
                        "n_classifiers": 5},
                       {"cons_class": DemographicParity, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.354174, "n_oracle_calls": 22,
                        "n_classifiers": 5},
                       {"cons_class": DemographicParity, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.365130, "n_oracle_calls": 22,
                        "n_classifiers": 5},
                       {"cons_class": EqualizedOdds, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.309333, "n_oracle_calls": 21,
                        "n_classifiers": 4},
                       {"cons_class": EqualizedOdds, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.378827, "n_oracle_calls": 19,
                           "n_classifiers": 6},
                       {"cons_class": EqualizedOdds, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.421531, "n_oracle_calls": 19,
                           "n_classifiers": 6},
                       {"cons_class": EqualizedOdds, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.435765, "n_oracle_calls": 19,
                           "n_classifiers": 6},
                       {"cons_class": EqualizedOdds, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.442883, "n_oracle_calls": 19,
                        "n_classifiers": 6}]

    def run_smoke_test(self, data):
        expgrad = ExponentiatedGradient(self.learner, constraints=data["cons_class"](),
                                        eps=data["eps"])
        expgrad.fit(self.X, self.y, sensitive_features=self.A)

        res = expgrad._expgrad_result._as_dict()
        Q = res["best_classifier"]
        res["n_classifiers"] = len(res["classifiers"])

        disp = data["cons_class"]()
        disp.load_data(self.X, self.y, sensitive_features=self.A)
        error = ErrorRate()
        error.load_data(self.X, self.y, sensitive_features=self.A)
        res["disp"] = disp.gamma(Q).max()
        res["error"] = error.gamma(Q)[0]

        assert res["best_gap"] == pytest.approx(
            data["best_gap"], abs=self._PRECISION)
        assert res["last_t"] == data["last_t"]
        assert res["best_t"] == data["best_t"]
        assert res["disp"] == pytest.approx(data["disp"], abs=self._PRECISION)
        assert res["error"] == pytest.approx(
            data["error"], abs=self._PRECISION)
        assert res["n_oracle_calls"] == data["n_oracle_calls"]
        assert res["n_classifiers"] == data["n_classifiers"]

    @pytest.mark.parametrize("testdata", smoke_test_data)
    def test_smoke(self, testdata):
        self.run_smoke_test(testdata)

    def test_simple_fit_predict(self):
        estimator = LeastSquaresBinaryClassifierLearner()
        constraints = DemographicParity()
        expgrad = ExponentiatedGradient(estimator, constraints)
        expgrad.fit(pd.DataFrame(X1), pd.Series(labels),
                    sensitive_features=pd.Series(sensitive_features))
        expgrad.predict(pd.DataFrame(X1))
