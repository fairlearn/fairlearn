# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import fairlearn.moments as moments
import fairlearn.classred as red

import pytest


class LeastSquaresLearner:
    def __init__(self):
        self.weights = None

    def fit(self, X, Y, W):
        sqrtW = np.sqrt(W)
        matX = np.array(X) * sqrtW[:, np.newaxis]
        vecY = Y * sqrtW
        self.lsqinfo = np.linalg.lstsq(matX, vecY, rcond=-1)
        self.weights = pd.Series(self.lsqinfo[0], index=list(X))

    def predict(self, X):
        pred = X.dot(self.weights)
        return 1 * (pred > 0.5)


class TestFairLearnSmoke:
    def setup_method(self, method):
        print("setup_method      method:%s" % method.__name__)
        attrs = [str(x) for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
        labls = [int(x) for x in '0110100' '0010111' '001111']
        feat1 = [int(x) for x in '0110101' '0111101' '001011']
        feat2 = [int(x) for x in '0000100' '0000011' '111111']
        feat3 = [int(x) for x in '1111111' '1111111' '111111']
        self.dataX = pd.DataFrame(
            {"feat1": feat1, "feat2": feat2, "feat3": feat3})
        self.dataY = pd.Series(labls)
        self.dataA = pd.Series(attrs)
        self.learner = LeastSquaresLearner()
        self._PRECISION = 1e-6

    smoke_test_data = [{"cons_class": moments.DP, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.250000, "n_oracle_calls": 32,
                        "n_classifiers": 3},
                       {"cons_class": moments.DP, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.266522, "n_oracle_calls": 23,
                        "n_classifiers": 6},
                       {"cons_class": moments.DP, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.332261, "n_oracle_calls": 22,
                        "n_classifiers": 5},
                       {"cons_class": moments.DP, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.354174, "n_oracle_calls": 22,
                        "n_classifiers": 5},
                       {"cons_class": moments.DP, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.365130, "n_oracle_calls": 22,
                        "n_classifiers": 5},
                       {"cons_class": moments.EO, "eps": 0.100,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.100000,
                        "error": 0.309333, "n_oracle_calls": 21,
                        "n_classifiers": 4},
                       {"cons_class": moments.EO, "eps": 0.050,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.050000,
                        "error": 0.378827, "n_oracle_calls": 19,
                           "n_classifiers": 6},
                       {"cons_class": moments.EO, "eps": 0.020,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.020000,
                        "error": 0.421531, "n_oracle_calls": 19,
                           "n_classifiers": 6},
                       {"cons_class": moments.EO, "eps": 0.010,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.010000,
                        "error": 0.435765, "n_oracle_calls": 19,
                           "n_classifiers": 6},
                       {"cons_class": moments.EO, "eps": 0.005,
                        "best_gap": 0.000000, "last_t": 5,
                        "best_t": 5, "disp": 0.005000,
                        "error": 0.442883, "n_oracle_calls": 19,
                        "n_classifiers": 6}, ]

    def run_smoke_test(self, data):
        res_tuple = red.expgrad(self.dataX, self.dataA, self.dataY,
                                self.learner,
                                constraints=data["cons_class"](),
                                eps=data["eps"])

        res = res_tuple._asdict()
        Q = res["best_classifier"]
        res["n_classifiers"] = len(res["classifiers"])

        disp = data["cons_class"]()
        disp.init(self.dataX, self.dataA, self.dataY)
        error = moments.MisclassificationError()
        error.init(self.dataX, self.dataA, self.dataY)
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
