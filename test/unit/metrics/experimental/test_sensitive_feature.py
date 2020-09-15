# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

import fairlearn.metrics.experimental as metrics


raw_feature = ['a', 'b', 'c', 'a']
expected_classes = ['a', 'b', 'c']


def test_list():
    target = metrics._sensitive_feature.SensitiveFeature(raw_feature, 0)

    assert target.name == "SF 0"
    assert np.array_equal(expected_classes, target.classes)

    mask = target.get_mask_for_class('a')
    assert np.array_equal(mask,  [True, False, False, True])
    mask_b = target.get_mask_for_class('b')
    assert np.array_equal(mask_b, [False, True, False, False])
    mask_c = target.get_mask_for_class('c')
    assert np.array_equal(mask_c, [False, False, True, False])
