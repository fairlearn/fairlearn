# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing._threshold_optimizer import _SUPPORTED_CONSTRAINTS
from fairlearn.postprocessing._constants import _MATPLOTLIB_IMPORT_ERROR_MESSAGE


class FakePredictor:
    def predict(self, X):
        return np.random.random(len(X))


@pytest.mark.parametrize("constraints", _SUPPORTED_CONSTRAINTS)
def test_no_matplotlib(constraints):
    n_samples = 50
    n_features = 50
    n_sensitive_feature_values = 2
    n_classes = 2

    threshold_optimizer = ThresholdOptimizer(unconstrained_predictor=FakePredictor(),
                                             constraints=constraints,
                                             plot=True)
    with pytest.raises(RuntimeError) as exc:
        threshold_optimizer.fit(X=np.random.random((n_samples, n_features)),
                                y=np.random.randint(n_classes, size=n_samples),
                                sensitive_features=np.random.randint(n_sensitive_feature_values,
                                                                     size=n_samples))
        assert str(exc.value) == _MATPLOTLIB_IMPORT_ERROR_MESSAGE
