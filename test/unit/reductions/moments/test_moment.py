# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from fairlearn.reductions import Moment
from fairlearn.reductions._moments.moment import _LABEL


def test_load_data_without_sensitive_features():
    moment = Moment()

    moment.load_data(np.array([[0], [1]]), np.array([0, 1]))

    assert moment.tags.to_native().columns.tolist() == [_LABEL]


class _SeriesMoment(Moment[pd.Series]):
    def gamma(self, predictor):
        return pd.Series([0.25])


class _ScalarMoment(Moment[float]):
    def gamma(self, predictor):
        return 0.25


def test_objective_value_from_series_result():
    assert _SeriesMoment().objective_value(lambda X: X) == 0.25


def test_objective_value_requires_override_for_non_series_result():
    with pytest.raises(TypeError, match="must implement objective_value"):
        _ScalarMoment().objective_value(lambda X: X)
