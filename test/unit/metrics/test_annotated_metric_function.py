# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

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


@pytest.mark.parametrize(
    ("kw_argument_mapping", "expected"), [({}, 0.6), ({"sample_weight": "weight"}, 0.75)]
)
def test_dataframe_types(constructor, kw_argument_mapping, expected):
    data = {
        "y_true": [0, 1, 1, 0, 1],
        "y_pred": [0, 0, 1, 0, 0],
        "weight": [1.0, 0.4, 0.4, 1.0, 0.4],
    }
    df_native = constructor(data)

    fc = AnnotatedMetricFunction(func=accuracy_score, kw_argument_mapping=kw_argument_mapping)
    assert fc(df_native) == expected


def test_dataframe_types_2d_array(constructor):
    data = {
        "y_true": [0, 1, 2, 0, 1, 2, 2, 1],
        "y_pred": [
            [0.5, 0.3, 0.2],
            [0.1, 0.5, 0.4],
            [0.1, 0.1, 0.8],
            [0.2, 0.5, 0.3],
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
            [0.2, 0.3, 0.5],
            [0.6, 0.1, 0.3],
        ],
    }

    df_native = constructor(data)
    func = functools.partial(roc_auc_score, multi_class="ovr", labels=[0, 1, 2])
    fc = AnnotatedMetricFunction(func=func, name="rod_auc_score")
    assert round(fc(df_native), 3) == 0.508
