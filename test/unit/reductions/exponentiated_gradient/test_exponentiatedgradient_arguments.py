# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import pytest


from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions import DemographicParity
from fairlearn.reductions import ErrorRate
from .simple_learners import LeastSquaresBinaryClassifierLearner
from .test_utilities import sensitive_features, X1, X2, X3, labels

from test.unit.input_convertors import conversions_for_1d, ensure_ndarray, ensure_dataframe

# ===============================================================

# Ways of transforming the data
candidate_X_transforms = [ensure_ndarray, ensure_dataframe]
candidate_Y_transforms = conversions_for_1d
candidate_A_transforms = conversions_for_1d

# ================================================================


class TestExponentiatedGradientArguments:
    def setup_method(self, method):
        self.X = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
        self.y = pd.Series(labels)
        self.A = pd.Series(sensitive_features)
        self.learner = LeastSquaresBinaryClassifierLearner()
        self._PRECISION = 1e-6

    @pytest.mark.parametrize("transformA", candidate_A_transforms)
    @pytest.mark.parametrize("transformY", candidate_Y_transforms)
    @pytest.mark.parametrize("transformX", candidate_X_transforms)
    def test_argument_types(self, transformX, transformY, transformA):
        # This is an expanded-out version of one of the smoke tests
        expgrad = ExponentiatedGradient(self.learner, constraints=DemographicParity(),
                                        eps=0.1)
        expgrad.fit(transformX(self.X), transformY(self.y), sensitive_features=transformA(self.A))

        res = expgrad._expgrad_result._as_dict()
        Q = res["best_classifier"]
        res["n_classifiers"] = len(res["classifiers"])

        disp = DemographicParity()
        disp.load_data(self.X, self.y, sensitive_features=self.A)
        error = ErrorRate()
        error.load_data(self.X, self.y, sensitive_features=self.A)
        res["disp"] = disp.gamma(Q).max()
        res["error"] = error.gamma(Q)[0]

        assert res["best_gap"] == pytest.approx(0.0000, abs=self._PRECISION)
        assert res["last_t"] == 5
        assert res["best_t"] == 5
        assert res["disp"] == pytest.approx(0.1, abs=self._PRECISION)
        assert res["error"] == pytest.approx(0.25, abs=self._PRECISION)
        assert res["n_oracle_calls"] == 32
        assert res["n_classifiers"] == 3
