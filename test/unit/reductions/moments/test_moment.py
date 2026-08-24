# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

from fairlearn.reductions import Moment
from fairlearn.reductions._moments.moment import _LABEL


def test_load_data_without_sensitive_features():
    moment = Moment()

    moment.load_data(np.array([[0], [1]]), np.array([0, 1]))

    assert moment.tags.to_native().columns.tolist() == [_LABEL]
