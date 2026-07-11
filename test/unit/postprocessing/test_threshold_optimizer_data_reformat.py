# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from fairlearn.postprocessing._constants import LABEL_KEY, SCORE_KEY, SENSITIVE_FEATURE_KEY
from fairlearn.postprocessing._threshold_optimizer import _reformat_and_group_data


def test_reformat_and_group_data_preserves_dataframe_sensitive_feature_columns():
    sensitive_features = pd.DataFrame({"SF1": ["a", "a", "b", "b"], "SF2": ["x", "y", "x", "y"]})

    grouped_data = _reformat_and_group_data(
        sensitive_features=sensitive_features,
        labels=[0, 1, 0, 1],
        scores=[0.1, 0.2, 0.3, 0.4],
    )

    assert list(grouped_data.obj.columns) == ["SF1", "SF2", SCORE_KEY, LABEL_KEY]
    assert sorted(grouped_data.groups.keys()) == [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]


def test_reformat_and_group_data_preserves_ndarray_sensitive_feature_columns():
    sensitive_features = np.array([["a", "x"], ["a", "y"], ["b", "x"], ["b", "y"]])

    grouped_data = _reformat_and_group_data(
        sensitive_features=sensitive_features,
        labels=[0, 1, 0, 1],
        scores=[0.1, 0.2, 0.3, 0.4],
    )

    assert list(grouped_data.obj.columns) == [
        f"{SENSITIVE_FEATURE_KEY}_0",
        f"{SENSITIVE_FEATURE_KEY}_1",
        SCORE_KEY,
        LABEL_KEY,
    ]
    assert sorted(grouped_data.groups.keys()) == [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]


def test_reformat_and_group_data_preserves_list_sensitive_feature_columns():
    sensitive_features = [["a", "x"], ["a", "y"], ["b", "x"], ["b", "y"]]

    grouped_data = _reformat_and_group_data(
        sensitive_features=sensitive_features,
        labels=[0, 1, 0, 1],
        scores=[0.1, 0.2, 0.3, 0.4],
        sensitive_feature_names=["first", "second"],
    )

    assert list(grouped_data.obj.columns) == ["first", "second", SCORE_KEY, LABEL_KEY]
    assert sorted(grouped_data.groups.keys()) == [("a", "x"), ("a", "y"), ("b", "x"), ("b", "y")]


def test_reformat_and_group_data_rejects_multi_column_scores():
    scores = pd.DataFrame({"score_1": [0.1, 0.2], "score_2": [0.3, 0.4]})

    with pytest.raises(ValueError, match="single column in score"):
        _reformat_and_group_data(
            sensitive_features=["a", "b"],
            labels=[0, 1],
            scores=scores,
        )


def test_reformat_and_group_data_rejects_reserved_sensitive_feature_names():
    sensitive_features = pd.DataFrame({SCORE_KEY: ["a", "b"]})

    with pytest.raises(ValueError, match="sensitive feature named"):
        _reformat_and_group_data(
            sensitive_features=sensitive_features,
            labels=[0, 1],
            scores=[0.1, 0.2],
        )
