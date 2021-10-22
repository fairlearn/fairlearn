# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import numpy as np
from sklearn.metrics import recall_score
from fairlearn.metrics._function_container import FunctionContainer


def test_constructor_unnamed():
    sample_params = {'sample_weight': [1, 2, 3]}

    fc = FunctionContainer(recall_score, None, sample_params)
    assert fc.name_ == recall_score.__name__
    assert len(fc.sample_params_) == 1
    assert isinstance(fc.sample_params_['sample_weight'], np.ndarray)
    assert np.array_equal(fc.sample_params_['sample_weight'], [1, 2, 3])


def test_constructor_no_name():
    # Tests case where no name is given and the function has no __name__
    sample_params = {'sample_weight': [1, 2, 3]}
    my_func = functools.partial(recall_score, pos_label=0)

    fc = FunctionContainer(my_func, None, sample_params)
    assert fc.name_ == 'metric'
    assert len(fc.sample_params_) == 1
    assert isinstance(fc.sample_params_['sample_weight'], np.ndarray)
    assert np.array_equal(fc.sample_params_['sample_weight'], [1, 2, 3])


def test_constructor_named():
    sample_params = {'sample_weight': [1, 2, 3]}

    fc = FunctionContainer(recall_score, "OverrideName", sample_params)
    assert fc.name_ == "OverrideName"
    assert len(fc.sample_params_) == 1
    assert isinstance(fc.sample_params_['sample_weight'], np.ndarray)
    assert np.array_equal(fc.sample_params_['sample_weight'], [1, 2, 3])
