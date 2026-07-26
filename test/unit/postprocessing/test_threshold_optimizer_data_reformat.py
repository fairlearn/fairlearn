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
    expected_group_data = {
        ("a", "x"): (0.1, 0),
        ("a", "y"): (0.2, 1),
        ("b", "x"): (0.3, 0),
        ("b", "y"): (0.4, 1),
    }
    for group_key, (score, label) in expected_group_data.items():
        group = grouped_data.get_group(group_key)
        assert group[SCORE_KEY].tolist() == [score]
        assert group[LABEL_KEY].tolist() == [label]


def test_reformat_and_group_data_preserves_dataframe_sensitive_feature_dtypes():
    sensitive_features = pd.DataFrame({"SF1": [0.1, 0.1], "SF2": [2**53, 2**53 + 1]})

    grouped_data = _reformat_and_group_data(
        sensitive_features=sensitive_features,
        labels=[0, 1],
        scores=[0.1, 0.2],
    )

    assert sorted(grouped_data.groups.keys()) == [
        (0.1, 2**53),
        (0.1, 2**53 + 1),
    ]


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


def test_reformat_and_group_data_preserves_tuple_sensitive_feature_columns():
    sensitive_features = [("a", "x"), ("b", "y")]

    grouped_data = _reformat_and_group_data(
        sensitive_features=sensitive_features,
        labels=[0, 1],
        scores=[0.1, 0.2],
        sensitive_feature_names=["first", "second"],
    )

    assert list(grouped_data.obj.columns) == ["first", "second", SCORE_KEY, LABEL_KEY]
    assert sorted(grouped_data.groups.keys()) == [("a", "x"), ("b", "y")]


@pytest.mark.parametrize(
    "input_name,input_value,error_match",
    [
        (
            "scores",
            pd.DataFrame({"score_1": [0.1, 0.2], "score_2": [0.3, 0.4]}),
            "single column in score",
        ),
        (
            "labels",
            pd.DataFrame({"label_1": [0, 1], "label_2": [1, 0]}),
            "single column in label",
        ),
    ],
)
def test_reformat_and_group_data_rejects_multi_column_auxiliary_data(
    input_name,
    input_value,
    error_match,
):
    inputs = {
        "sensitive_features": ["a", "b"],
        "labels": [0, 1],
        "scores": [0.1, 0.2],
    }
    inputs[input_name] = input_value

    with pytest.raises(ValueError, match=error_match):
        _reformat_and_group_data(**inputs)


def test_reformat_and_group_data_rejects_singleton_score_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        _reformat_and_group_data(
            sensitive_features=["a", "a", "b", "b"],
            labels=[0, 1, 0, 1],
            scores=[0.5],
        )


@pytest.mark.parametrize(
    "sensitive_features",
    [
        [("a", "x"), ("b",)],
        [[["a"]], [["b"]]],
        np.array([[["a"]], [["b"]]]),
    ],
)
def test_reformat_and_group_data_rejects_malformed_nested_sensitive_features(
    sensitive_features,
):
    with pytest.raises(ValueError, match="single column in sensitive_feature"):
        _reformat_and_group_data(
            sensitive_features=sensitive_features,
            labels=[0, 1],
            scores=[0.1, 0.2],
        )


@pytest.mark.parametrize(
    "sensitive_features,sensitive_feature_names,error_match",
    [
        (
            [["a", "x"], ["b", "y"]],
            ["first"],
            "column names need to be of equal length",
        ),
        (
            pd.DataFrame([["a", "x"], ["b", "y"]], columns=["group", "group"]),
            None,
            "column names need to be unique",
        ),
        (
            [["a", "x"], ["b", "y"]],
            ["group", "group"],
            "column names need to be unique",
        ),
    ],
)
def test_reformat_and_group_data_rejects_invalid_sensitive_feature_names(
    sensitive_features,
    sensitive_feature_names,
    error_match,
):
    with pytest.raises(ValueError, match=error_match):
        _reformat_and_group_data(
            sensitive_features=sensitive_features,
            labels=[0, 1],
            scores=[0.1, 0.2],
            sensitive_feature_names=sensitive_feature_names,
        )


def test_reformat_and_group_data_rejects_reserved_sensitive_feature_names():
    sensitive_features = pd.DataFrame({SCORE_KEY: ["a", "b"]})

    with pytest.raises(ValueError, match="sensitive feature named"):
        _reformat_and_group_data(
            sensitive_features=sensitive_features,
            labels=[0, 1],
            scores=[0.1, 0.2],
        )
