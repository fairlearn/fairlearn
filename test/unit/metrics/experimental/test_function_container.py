# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.metrics import recall_score
from fairlearn.metrics.experimental._function_container import FunctionContainer


def test_constructor_unnamed():
    sample_param_names = ['sample_weight']
    params = {'pos_label': 0, 'sample_weight': [1, 2, 3]}

    fc = FunctionContainer(recall_score, None, sample_param_names, params)
    assert fc.name == recall_score.__name__
    assert len(fc.sample_param_names) == 1
    assert fc.sample_param_names[0] == 'sample_weight'
    assert len(fc.params) == 2
    assert fc.params['pos_label'] == 0
    assert isinstance(fc.params['sample_weight'], np.ndarray)
    assert np.array_equal(fc.params['sample_weight'], [1, 2, 3])


def test_constructor_named():
    sample_param_names = ['sample_weight']
    params = {'pos_label': 0, 'sample_weight': [1, 2, 3]}

    fc = FunctionContainer(recall_score, "OverrideName", sample_param_names, params)
    assert fc.name == "OverrideName"
    assert len(fc.sample_param_names) == 1
    assert fc.sample_param_names[0] == 'sample_weight'
    assert len(fc.params) == 2
    assert fc.params['pos_label'] == 0
    assert isinstance(fc.params['sample_weight'], np.ndarray)
    assert np.array_equal(fc.params['sample_weight'], [1, 2, 3])


