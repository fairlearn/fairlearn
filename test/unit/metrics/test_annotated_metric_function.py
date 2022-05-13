# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools

import numpy as np
from sklearn.metrics import recall_score

from fairlearn.metrics._annotated_metric_function import AnnotatedMetricFunction


def test_constructor_unnamed():
    fc = AnnotatedMetricFunction(func=recall_score, name=None)
    assert fc.name == recall_score.__name__
    assert np.array_equal(fc.postional_argument_names, ["y_true", "y_pred"])
    assert isinstance(fc.kw_argument_mapping, dict)
    assert len(fc.kw_argument_mapping) == 0


def test_constructor_no_name(recwarn):
    # Tests case where no name is given and the function has no __name__
    my_func = functools.partial(recall_score, pos_label=0)

    fc = AnnotatedMetricFunction(func=my_func, name=None)
    assert fc.name == "metric"
    assert np.array_equal(fc.postional_argument_names, ["y_true", "y_pred"])
    assert isinstance(fc.kw_argument_mapping, dict)
    assert len(fc.kw_argument_mapping) == 0
    assert len(recwarn) == 1
    assert str(recwarn[0].message) == "Supplied 'func' had no __name__ attribute"


def test_constructor_named():
    fc = AnnotatedMetricFunction(func=recall_score, name="OverrideName")
    assert fc.name == "OverrideName"
    assert np.array_equal(fc.postional_argument_names, ["y_true", "y_pred"])
    assert isinstance(fc.kw_argument_mapping, dict)
    assert len(fc.kw_argument_mapping) == 0
