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


def test_generate_params_for_mask():
    sample_param_names = ['sample_weight']
    params = {'pos_label': 0, 'sample_weight': [1, 2, 3]}

    fc = FunctionContainer(recall_score, None, sample_param_names, params)

    mask = [True, False, True]
    result = fc.generate_params_for_mask(mask)
    assert len(result) == 2
    assert result['pos_label'] == 0
    assert isinstance(result['sample_weight'], np.ndarray)
    assert np.array_equal(result['sample_weight'], [1, 3])


def test_function_evaluation():
    sample_param_names = ['sample_weight']
    params = {'pos_label': 0, 'sample_weight': [1, 2, 3, 1, 1, 2, 2, 1]}

    fc = FunctionContainer(recall_score, None, sample_param_names, params)

    mask = [True, False, True, False, False, True, True, False]
    y_t = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    y_p = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    y_t_masked = [0, 0, 1, 0]
    y_p_masked = [0, 0, 1, 1]
    wgt_masked = [1, 3, 2, 2]

    expected = recall_score(y_t_masked, y_p_masked, pos_label=0, sample_weight=wgt_masked)
    actual = fc.evaluate(y_t, y_p, mask)
    assert expected == actual
