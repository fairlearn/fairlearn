# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.metrics import recall_score
from fairlearn.metrics.experimental._function_container import FunctionContainer


def test_constructor_unnamed():
    sample_params = {'sample_weight': [1, 2, 3]}
    params = {'pos_label': 0, }

    fc = FunctionContainer(recall_score, None, sample_params, params)
    assert fc.name_ == recall_score.__name__
    assert len(fc.sample_params_) == 1
    assert isinstance(fc.sample_params_['sample_weight'], np.ndarray)
    assert np.array_equal(fc.sample_params_['sample_weight'], [1, 2, 3])
    assert len(fc.params_) == 1
    assert fc.params_['pos_label'] == 0


def test_constructor_named():
    sample_params = {'sample_weight': [1, 2, 3]}
    params = {'pos_label': 0}

    fc = FunctionContainer(recall_score, "OverrideName", sample_params, params)
    assert fc.name_ == "OverrideName"
    assert len(fc.sample_params_) == 1
    assert isinstance(fc.sample_params_['sample_weight'], np.ndarray)
    assert np.array_equal(fc.sample_params_['sample_weight'], [1, 2, 3])
    assert len(fc.params_) == 1
    assert fc.params_['pos_label'] == 0


def test_generate_params_for_mask():
    sample_params = {'sample_weight': [1, 2, 3]}
    params = {'pos_label': 0}

    fc = FunctionContainer(recall_score, None, sample_params, params)

    mask = [True, False, True]
    result = fc.generate_params_for_mask(mask)
    assert len(result) == 2
    assert result['pos_label'] == 0
    assert isinstance(result['sample_weight'], np.ndarray)
    assert np.array_equal(result['sample_weight'], [1, 3])


def test_function_evaluation():
    sample_params = {'sample_weight': [1, 2, 3, 1, 1, 2, 2, 1]}
    params = {'pos_label': 0}

    fc = FunctionContainer(recall_score, None, sample_params, params)

    mask = [True, False, True, False, False, True, True, False]
    y_t = np.asarray([0, 1, 0, 1, 0, 1, 0, 1])
    y_p = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    y_t_masked = [0, 0, 1, 0]
    y_p_masked = [0, 0, 1, 1]
    wgt_masked = [1, 3, 2, 2]

    expected = recall_score(y_t_masked, y_p_masked,
                            pos_label=0,
                            sample_weight=wgt_masked)
    actual = fc.evaluate(y_t, y_p, mask)
    assert expected == actual

    expected_all = recall_score(y_t, y_p,
                                pos_label=0,
                                sample_weight=sample_params['sample_weight'])
    actual_all = fc.evaluate_all(y_t, y_p)
    assert expected_all == actual_all
