# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

from fairlearn.reductions._constant_predictor import ConstantPredictor


def test_smoke():
    const_value = 123
    cp = ConstantPredictor(const_value)

    assert cp.predict('a') == const_value
    assert cp.predict(1) == const_value
    assert cp.predict(1, 1) == const_value
    assert cp.predict(np.zeros(10)) == const_value
    assert cp.predict(X="a", sample_weights=[1, 2, 3]) == const_value
    assert cp.predict([1, 2, 3], sample_weights=[4, 5, 6]) == const_value
    assert cp.predict(c=np.zeros(10), d='a string', e=1.0) == const_value