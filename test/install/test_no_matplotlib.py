# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from sklearn.base import BaseEstimator, ClassifierMixin

from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from fairlearn.postprocessing._threshold_optimizer import SIMPLE_CONSTRAINTS
from fairlearn.postprocessing._constants import _MATPLOTLIB_IMPORT_ERROR_MESSAGE


class FakePredictor(BaseEstimator, ClassifierMixin):
    def fit(self, X, y=None, **kwargs):
        return self

    def predict(self, X):
        return np.random.random(len(X))


@pytest.mark.parametrize("constraints", [*SIMPLE_CONSTRAINTS, 'equalized_odds'])
def test_no_matplotlib(constraints):
    n_samples = 50
    n_features = 50
    n_sensitive_feature_values = 2
    n_classes = 2

    threshold_optimizer = ThresholdOptimizer(estimator=FakePredictor(),
                                             constraints=constraints)
    threshold_optimizer.fit(X=np.random.random((n_samples, n_features)),
                            y=np.random.randint(n_classes, size=n_samples),
                            sensitive_features=np.random.randint(n_sensitive_feature_values,
                                                                 size=n_samples))

    with pytest.raises(RuntimeError) as exc:
        plot_threshold_optimizer(threshold_optimizer)
        assert str(exc.value) == _MATPLOTLIB_IMPORT_ERROR_MESSAGE
