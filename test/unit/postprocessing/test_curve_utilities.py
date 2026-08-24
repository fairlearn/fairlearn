# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import narwhals.stable.v1 as nw
import numpy as np
import pytest

from fairlearn.postprocessing._constants import (
    LABEL_KEY,
    SCORE_KEY,
    SENSITIVE_FEATURE_KEY,
)
from fairlearn.postprocessing._threshold_operation import ThresholdOperation
from fairlearn.postprocessing._tradeoff_curve_utilities import (
    _calculate_tradeoff_points,
    _filter_points_to_get_convex_hull,
    _interpolate_curve,
    _tradeoff_curve,
)

from .conftest import (
    _get_grouped_data_and_base_points,
    labels_ex,
    scores_ex,
    sensitive_feature_names_ex1,
    sensitive_features_ex1,
)

pytestmark = pytest.mark.narwhals


def test_assert_interpolated_curve(constructor):
    # An easily interpretable test to make sure the assertion method works as expected
    base_points = constructor(
        {"x": [0, 5, 10], "y": [0.0, 2.5, 5.0], "operation": ["a", "b", "c"]}  # irrelevant
    )
    x_grid = np.linspace(0, 10, 333)
    curve = _interpolate_curve(base_points, "x", "y", "operation", x_grid)

    _assert_interpolated_points_are_between_base_points(base_points, curve)


def test_interpolate_curve(constructor):
    # The operation is irrelevant in this case since its semantics are not
    # used within _interpolate_curve.
    base_points = constructor(
        {
            "x": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "y": [-5.0, -2.0, -1.5, -1.0, 0.0, 0.5, 0.8, 1.0, 1.1, 1.15],
            "operation": ["i", "r", "r", "e", "l", "e", "v", "a", "n", "t"],
        }
    )
    x_grid = np.linspace(0, 9, 100)
    curve = _interpolate_curve(base_points, "x", "y", "operation", x_grid)

    _assert_interpolated_points_are_between_base_points(base_points, curve)


@pytest.mark.parametrize(
    "data,expected_remaining_indices",
    [
        ({"x": [0, 1], "y": [0, 1], "operation": ["i", "r"]}, [0, 1]),
        (
            {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.5, 1.0], "operation": ["i", "r", "e"]},
            [0, 2],
        ),
        (
            {"x": [0.0, 0.5, 1.0], "y": [0.0, 0.51, 1.0], "operation": ["i", "r", "e"]},
            [0, 1, 2],
        ),
        (
            {
                "x": [0.0, 0.2, 0.3, 0.5, 1.0],
                "y": [0.0, 0.3, 0.35, 0.7, 1.0],
                "operation": ["i", "r", "r", "e", "l"],
            },
            [0, 1, 3, 4],
        ),
        (
            {
                "x": [0.0, 0.1, 0.2, 0.5, 1.0],
                "y": [0.0, 0.3, 0.5, 0.9, 1.0],
                "operation": ["i", "r", "r", "e", "l"],
            },
            [0, 1, 2, 3, 4],
        ),
        (
            {
                "x": [0.0, 0.2, 0.3, 0.5, 1.0],
                "y": [0.0, 0.3, 0.8, 0.82, 1.0],
                "operation": ["i", "r", "r", "e", "l"],
            },
            [0, 2, 4],
        ),
    ],
)
def test_convex_hull(constructor, data, expected_remaining_indices):
    base_points = constructor(data)
    convex_hull = _filter_points_to_get_convex_hull(base_points)

    expected = nw.maybe_reset_index(
        nw.from_native(base_points, eager_only=True, pass_through=False)[
            expected_remaining_indices
        ]
    ).to_native()
    assert expected.equals(convex_hull)


def test_calculate_tradeoff_points(request, constructor):
    if "pyarrow_table" in str(request):
        reason = (
            "pyarrow.lib.ArrowInvalid: Could not convert [>inf] with type ThresholdOperation: "
            "did not recognize Python value type when inferring an Arrow data type"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    data = constructor(
        {
            SENSITIVE_FEATURE_KEY: sensitive_features_ex1.squeeze(),
            SCORE_KEY: scores_ex.squeeze(),
            LABEL_KEY: labels_ex.squeeze(),
        }
    )
    grouped_data = (
        nw.from_native(data, eager_only=True, pass_through=False)
        .filter(nw.col(SENSITIVE_FEATURE_KEY) == "A")
        .sort(by=SCORE_KEY, descending=True)
        .to_native()
    )

    roc_points = _calculate_tradeoff_points(grouped_data, "A", flip=True)
    expected_roc_points = constructor(
        {
            "x": [0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0],
            "y": [0.0, 0.0, 1 / 3, 0.0, 0.1 / 0.3, 2 / 3, 1, 0.2 / 0.3, 1.0, 1.0],
            "operation": [
                ThresholdOperation(">", np.inf),
                ThresholdOperation("<", -np.inf),
                ThresholdOperation("<", 0.5),
                ThresholdOperation(">", 2.5),
                ThresholdOperation(">", 1.5),
                ThresholdOperation("<", 1.5),
                ThresholdOperation("<", 2.5),
                ThresholdOperation(">", 0.5),
                ThresholdOperation("<", np.inf),
                ThresholdOperation(">", -np.inf),
            ],
        }
    )

    _assert_equal_points(expected_roc_points, roc_points)

    expected_roc_convex_hull = constructor(
        {
            "x": [0.0, 0.5, 1.0],
            "y": [0, 1, 1],
            "operation": [
                ThresholdOperation(">", np.inf),
                ThresholdOperation("<", 2.5),
                ThresholdOperation(">", -np.inf),
            ],
        }
    )
    # Try filtering to get the convex hull of the ROC points.
    selected_points = constructor(_filter_points_to_get_convex_hull(roc_points))[
        ["x", "y", "operation"]
    ]
    _assert_equal_points(expected_roc_convex_hull, selected_points)


def test_tradeoff_curve(request, constructor):
    if "pyarrow_table" in str(request):
        reason = (
            "pyarrow.lib.ArrowInvalid: Could not convert [>inf] with type ThresholdOperation: "
            "did not recognize Python value type when inferring an Arrow data type"
        )
        request.applymarker(pytest.mark.xfail(reason=reason))

    for sensitive_feature_value in sensitive_feature_names_ex1:
        (
            grouped_data,
            base_points,
            ignore_for_base_points,
            x_grid,
        ) = _get_grouped_data_and_base_points(constructor, sensitive_feature_value)

        roc_convex_hull = _tradeoff_curve(grouped_data, x_grid, sensitive_feature_value)
        curve = _interpolate_curve(roc_convex_hull, "x", "y", "operation", x_grid)

        _assert_interpolated_points_are_between_base_points(
            base_points, curve, ignore_for_base_points
        )


def _assert_interpolated_points_are_between_base_points(
    base_points, curve, ignore_for_base_points=None
):
    def _get_base_point_coordinates(i: int, data: nw.DataFrame):
        return data.item(i, "x"), data.item(i, "y")

    curve = nw.from_native(curve, eager_only=True)
    base_points = nw.from_native(base_points, eager_only=True)

    if ignore_for_base_points is None:
        ignore_for_base_points = []

    # Determine base point indices from base points such that points which are
    # not corners on the convex hull are ignored. These points are listed by
    # index in ignore_for_base_points.
    start_base_point_index = 0
    while start_base_point_index in ignore_for_base_points:
        start_base_point_index += 1

    base_point_index = start_base_point_index + 1
    while base_point_index in ignore_for_base_points:
        base_point_index += 1

    current_base_point_x, current_base_point_y = _get_base_point_coordinates(
        start_base_point_index, base_points
    )
    next_base_point_x, next_base_point_y = _get_base_point_coordinates(
        base_point_index, base_points
    )

    for x_grid_index in range(len(curve)):
        x = curve.item(x_grid_index, "x")
        y = curve.item(x_grid_index, "y")
        if np.isclose(x, current_base_point_x):
            assert np.isclose(y, current_base_point_y)
            continue

        while x > next_base_point_x:
            current_base_point_x, current_base_point_y = _get_base_point_coordinates(
                base_point_index, base_points
            )
            base_point_index += 1
            while base_point_index in ignore_for_base_points:
                base_point_index += 1
            next_base_point_x, next_base_point_y = _get_base_point_coordinates(
                base_point_index, base_points
            )

        if np.isclose(x, current_base_point_x):
            assert np.isclose(y, current_base_point_y)
            continue

        if np.isclose(x, next_base_point_x):
            assert np.isclose(y, next_base_point_y)
            continue

        # We know that current_base_point_x < x < next_base_point_x.
        # Ensure that the curve point lies exactly between the two base points
        # by checking the slope of the lines connecting the curve point to the
        # base points.
        assert np.isclose(
            (y - current_base_point_y) / (x - current_base_point_x),
            (next_base_point_y - y) / (next_base_point_x - x),
        )


def _assert_equal_points(expected_points, actual_points):
    nw_expected_points = nw.from_native(expected_points, eager_only=True, pass_through=False)
    nw_actual_points = nw.from_native(actual_points, eager_only=True, pass_through=False)

    assert len(nw_expected_points) == len(nw_actual_points)

    for i in range(len(expected_points)):
        assert np.isclose(nw_actual_points.item(i, "x"), nw_expected_points.item(i, "x"))
        assert np.isclose(nw_actual_points.item(i, "y"), nw_expected_points.item(i, "y"))
        assert (
            nw_actual_points.item(i, "operation").operator
            == nw_expected_points.item(i, "operation").operator
        )
        assert np.isclose(
            nw_actual_points.item(i, "operation").threshold,
            nw_expected_points.item(i, "operation").threshold,
        )
