from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairlearn.metrics._bootstrap import _align_sample_indices


@pytest.mark.parametrize(
    ["samples", "expected"],
    [
        (
            [
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "b", "c"],
                        "metric_1": [1, 2, 3],
                        "metric_2": [4, 5, 6],
                    },
                ).set_index("sensitive_feature_0"),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "b", "c"],
                        "metric_1": [10, 20, 30],
                        "metric_2": [40, 50, 60],
                    },
                ).set_index("sensitive_feature_0"),
            ],
            [
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "b", "c"],
                        "metric_1": [1, 2, 3],
                        "metric_2": [4, 5, 6],
                    },
                ).set_index("sensitive_feature_0"),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "b", "c"],
                        "metric_1": [10, 20, 30],
                        "metric_2": [40, 50, 60],
                    },
                ).set_index("sensitive_feature_0"),
            ],
        ),
        (
            [
                pd.DataFrame(
                    {"sensitive_feature_0": ["a"], "metric_1": [1], "metric_2": [4]},
                ).set_index("sensitive_feature_0"),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["b", "c"],
                        "metric_1": [20, 30],
                        "metric_2": [50, 60],
                    },
                ).set_index("sensitive_feature_0"),
                pd.DataFrame(
                    {"sensitive_feature_0": ["c"], "metric_1": [10], "metric_2": [40]},
                ).set_index("sensitive_feature_0"),
            ],
            [
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "b", "c"],
                        "metric_1": [1, np.nan, np.nan],
                        "metric_2": [4, np.nan, np.nan],
                    },
                ).set_index("sensitive_feature_0"),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "b", "c"],
                        "metric_1": [np.nan, 20, 30],
                        "metric_2": [np.nan, 50, 60],
                    },
                ).set_index("sensitive_feature_0"),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "b", "c"],
                        "metric_1": [np.nan, np.nan, 10],
                        "metric_2": [np.nan, np.nan, 40],
                    },
                ).set_index("sensitive_feature_0"),
            ],
        ),
        (
            [
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "a"],
                        "control_feature_0": ["x", "y"],
                        "metric_1": [1, 2],
                        "metric_2": [4, 5],
                    },
                ).set_index(["sensitive_feature_0", "control_feature_0"]),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["b", "c"],
                        "control_feature_0": ["x", "z"],
                        "metric_1": [20, 30],
                        "metric_2": [50, 60],
                    },
                ).set_index(["sensitive_feature_0", "control_feature_0"]),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["c", "c"],
                        "control_feature_0": ["y", "z"],
                        "metric_1": [10, 30],
                        "metric_2": [40, 60],
                    },
                ).set_index(["sensitive_feature_0", "control_feature_0"]),
            ],
            [
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "a", "b", "c", "c"],
                        "control_feature_0": ["x", "y", "x", "y", "z"],
                        "metric_1": [1, 2, np.nan, np.nan, np.nan],
                        "metric_2": [4, 5, np.nan, np.nan, np.nan],
                    },
                ).set_index(["sensitive_feature_0", "control_feature_0"]),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "a", "b", "c", "c"],
                        "control_feature_0": ["x", "y", "x", "y", "z"],
                        "metric_1": [np.nan, np.nan, 20, np.nan, 30],
                        "metric_2": [np.nan, np.nan, 50, np.nan, 60],
                    },
                ).set_index(["sensitive_feature_0", "control_feature_0"]),
                pd.DataFrame(
                    {
                        "sensitive_feature_0": ["a", "a", "b", "c", "c"],
                        "control_feature_0": ["x", "y", "x", "y", "z"],
                        "metric_1": [np.nan, np.nan, np.nan, 10, 30],
                        "metric_2": [np.nan, np.nan, np.nan, 40, 60],
                    },
                ).set_index(["sensitive_feature_0", "control_feature_0"]),
            ],
        ),
    ],
)
def test_align_sample_indices(samples: list[pd.DataFrame], expected: list[pd.DataFrame]) -> None:
    aligned_samples = _align_sample_indices(samples=samples)
    for aligned, exp in zip(aligned_samples, expected):
        pd.testing.assert_frame_equal(aligned, exp)
