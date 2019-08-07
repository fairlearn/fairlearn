# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import print_function

import functools
import numpy as np
import pandas as pd
import fairlearn.moments as moments
import fairlearn.classred as red

print = functools.partial(print, flush=True)


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
        return 1*(pred > 0.5)


tests = [{"constraints_class": moments.DemographicParity, "eps": 0.100, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.100000, "error": 0.250000,
          "n_oracle_calls":  32, "n_classifiers": 3},
         {"constraints_class": moments.DemographicParity, "eps": 0.050, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.050000,
          "error": 0.266522, "n_oracle_calls":  23, "n_classifiers": 6},
         {"constraints_class": moments.DemographicParity, "eps": 0.020, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.020000, "error": 0.332261,
          "n_oracle_calls":  22, "n_classifiers": 5},
         {"constraints_class": moments.DemographicParity, "eps": 0.010, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.010000, "error": 0.354174,
          "n_oracle_calls":  22, "n_classifiers": 5},
         {"constraints_class": moments.DemographicParity, "eps": 0.005, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.005000, "error": 0.365130,
          "n_oracle_calls":  22, "n_classifiers": 5},
         {"constraints_class": moments.EqualizedOdds, "eps": 0.100, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.100000, "error": 0.309333,
          "n_oracle_calls":  21, "n_classifiers": 4},
         {"constraints_class": moments.EqualizedOdds, "eps": 0.050, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.050000, "error": 0.378827,
          "n_oracle_calls":  19, "n_classifiers": 6},
         {"constraints_class": moments.EqualizedOdds, "eps": 0.020, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.020000, "error": 0.421531,
          "n_oracle_calls":  19, "n_classifiers": 6},
         {"constraints_class": moments.EqualizedOdds, "eps": 0.010, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.010000, "error": 0.435765,
          "n_oracle_calls":  19, "n_classifiers": 6},
         {"constraints_class": moments.EqualizedOdds, "eps": 0.005, "best_gap": 0.000000,
          "last_t": 5, "best_t": 5, "disp": 0.005000, "error": 0.442883,
          "n_oracle_calls":  19, "n_classifiers": 6},
         ]

_PRECISION = 1e-6


def test_res_float(key, res, test, report_list):
    if abs(res[key]-test[key]) > _PRECISION:
        report_list.append("%s_diff=%e" % (key, res[key]-test[key]))


def test_res_int(key, res, test, report_list):
    if abs(res[key]-test[key]) > 0:
        report_list.append("%s_diff=%d" % (key, res[key]-test[key]))


if __name__ == '__main__':
    attrs = [str(x) for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
    labls = [int(x) for x in '0110100' '0010111' '001111']
    feat1 = [int(x) for x in '0110101' '0111101' '001011']
    feat2 = [int(x) for x in '0000100' '0000011' '111111']
    feat3 = [int(x) for x in '1111111' '1111111' '111111']

    dataX = pd.DataFrame({"feat1": feat1, "feat2": feat2, "feat3": feat3})
    dataY = pd.Series(labls)
    dataA = pd.Series(attrs)

    learner = LeastSquaresLearner()

    for test in tests:
        res_tuple = red.exponentiated_gradient_reduction(dataX, dataA, dataY, learner,
                                constraints=test["constraints_class"](), eps=test["eps"])
        res = res_tuple._asdict()
        Q = res["best_classifier"]
        res["n_classifiers"] = len(res["classifiers"])

        disp = test["constraints_class"]()
        disp.init(dataX, dataA, dataY)

        error = moments.MisclassError()
        error.init(dataX, dataA, dataY)

        res["disp"] = disp.gamma(Q).max()
        res["error"] = error.gamma(Q)[0]
        report_header = "testing (%s, eps=%.3f)" \
                        % (test["constraints_class"].short_name, test["eps"])
        report_list = []
        test_res_float("best_gap", res, test, report_list)
        test_res_int("last_t", res, test, report_list)
        test_res_int("best_t", res, test, report_list)
        test_res_float("disp", res, test, report_list)
        test_res_float("error", res, test, report_list)
        test_res_int("n_oracle_calls", res, test, report_list)
        test_res_int("n_classifiers", res, test, report_list)
        if report_list:
            print("%s: %s" % (report_header, ", ".join(report_list)))
        else:
            print("%s: ok" % report_header)
