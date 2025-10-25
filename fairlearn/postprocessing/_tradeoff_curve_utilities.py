# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import Literal, Tuple

import narwhals.stable.v1 as nw
import numpy as np
from narwhals.typing import IntoDataFrame, IntoDataFrameT
from numpy.typing import NDArray
from sklearn.utils import Bunch

from ._constants import LABEL_KEY, SCORE_KEY
from ._threshold_operation import ThresholdOperation

DEGENERATE_LABELS_ERROR_MESSAGE = "Degenerate labels for sensitive feature value {}"

# Dictionary of metrics based on confusion matrix. Their input must be a Bunch with the fields
# named n, positives, negatives, predicted_positives, predicted_negatives, true_positives,
# true_negatives, false_positives, false_negatives. The fields indicate the counts. They can all
# be numpy arrays of the same length. Metrics are expected to return NaN where undefined.
METRIC_DICT = {
    "selection_rate": lambda x: x.predicted_positives / x.n,
    "false_positive_rate": lambda x: x.false_positives / x.negatives,
    "false_negative_rate": lambda x: x.false_negatives / x.positives,
    "true_positive_rate": lambda x: x.true_positives / x.positives,
    "true_negative_rate": lambda x: x.true_negatives / x.negatives,
    "accuracy_score": lambda x: (x.true_positives + x.true_negatives) / x.n,
    "balanced_accuracy_score": (
        lambda x: 0.5 * x.true_positives / x.positives + 0.5 * x.true_negatives / x.negatives
    ),
}


def _extend_confusion_matrix(*, true_positives, false_positives, true_negatives, false_negatives):
    """Extend the provided confusion matrix counts with additional implied fields.

    Parameters
    ----------
    true_positives, false_positives, true_negatives, false_negatives : int
        The counts appearing in the confusion matrix.

    Returns
    -------
    result : sklearn.utils.Bunch
        Dictionary-like object, with attributes:

        true_positives, false_positives, true_negatives, false_negatives : int
            The provided counts.

        predicted_positives, predicted_negatives, positives, negatives, n : int
            Derived counts.
    """
    return Bunch(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
        predicted_positives=(true_positives + false_positives),
        predicted_negatives=(true_negatives + false_negatives),
        positives=(true_positives + false_negatives),
        negatives=(true_negatives + false_positives),
        n=(true_positives + true_negatives + false_positives + false_negatives),
    )


def _tradeoff_curve(
    data: IntoDataFrameT,
    sensitive_feature_value,
    flip: bool = False,
    x_metric: Literal[
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "true_negative_rate",
        "false_negative_rate",
    ] = "false_positive_rate",
    y_metric: Literal[
        "accuracy_score",
        "balanced_accuracy_score",
        "selection_rate",
        "true_positive_rate",
        "true_negative_rate",
    ] = "true_positive_rate",
) -> IntoDataFrameT:
    """Get a convex hull of achievable trade-offs between the two provided metrics.

    The metrics are based on considering all possible thresholds of 'score' column of `data` and
    evaluated with respect to 'label' column of `data`.

    Parameters
    ----------
    data : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        Data frame with columns 'score' and 'label'.

    sensitive_feature_value : str or int
        The sensitive feature value of the samples providing in `data`. Only used
        to generate a description when an exception is thrown.

    flip : bool, default=False
        If True, also consider the flipped thresholding (points below the threshold
        classified as positive and above the threshold as negative).

    x_metric : str, default="false_positive_rate"
        The metric to use for the x-axis. Possible values are:
        "selection_rate", "true_positive_rate", "false_positive_rate",
        "true_negative_rate", "false_negative_rate".

    y_metric : str, default="true_positive_rate"
        The metric to use for the y-axis. Possible values are:
        "accuracy_score", "balanced_accuracy_score", "selection_rate",
        "true_positive_rate", "true_negative_rate".

    Returns
    -------
    result : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        The convex hull over the achievable trade-off points with columns
        'x', 'y', and 'operation'.
    """
    nw_data = nw.from_native(data, eager_only=True, pass_through=False)
    points_sorted = _calculate_tradeoff_points(
        nw_data, sensitive_feature_value, flip=flip, x_metric=x_metric, y_metric=y_metric
    )
    return _filter_points_to_get_convex_hull(points_sorted)


def _filter_points_to_get_convex_hull(points_sorted: IntoDataFrameT) -> IntoDataFrameT:
    """Find the upper convex hull.

    Parameters
    ----------
    points_sorted : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        Points represented as rows with 'x' and 'y' columns, sorted lexicographically by 'x' and 'y'.

    Returns
    -------
    result : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        Points that make the upper convex hull.

    Notes
    -----
    Uses Andrew's monotone chain convex hull algorithm :footcite:`wikibooks2010monotone`
    """
    selected_rows, selected_indexes = [], []
    output_cols = ("x", "y", "operation")
    nw_points_sorted = nw.from_native(points_sorted, eager_only=True).select(*output_cols)

    for _idx, r2 in enumerate(nw_points_sorted.iter_rows(named=True)):
        # For each set of three points, i.e. the last two points in selected
        # and the next point from the sorted list of base points, check
        # whether the middle point (r1) lies above the line between the
        # first point (r0) and the next base point (r2). If it is above,
        # it is indeed required for the convex hull. If it is below or
        # on the line, then it is part of the convex hull as defined with
        # just r0 and r2 and we can drop it from the list of selected points.
        while len(selected_rows) >= 2:
            r1 = selected_rows[-1]
            r0 = selected_rows[-2]
            # Compare slopes of lines between r0 and r1/r2 to determine
            # whether or not to drop r1. Instead of delta_y/delta_x we
            # multiplied both sides of the inequation by the delta_xs.
            if (r1["y"] - r0["y"]) * (r2["x"] - r0["x"]) <= (r2["y"] - r0["y"]) * (
                r1["x"] - r0["x"]
            ):
                # drop r1
                selected_rows.pop()
                selected_indexes.pop()
            else:
                break
        selected_rows.append(r2)
        selected_indexes.append(_idx)

    # The reason for this workaround is to be able to maintain the original dtype_backend
    # in case of pandas DataFrame's.
    return nw.maybe_reset_index(nw_points_sorted[selected_indexes, :]).to_native()


def _interpolate_curve(
    data: IntoDataFrameT, x_col: str, y_col: str, content_col: str, x_grid: NDArray
) -> IntoDataFrameT:
    """Interpolates the DataFrame in `data` along the values in `x_grid`.

    Assumes: (1) data[y_col] is concave in data[x_col]
             (2) min and max in x_grid are above and below min and max in data[x_col], respectively

    Parameters
    ----------
    data : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        The convex hull data points.
    x_col : str
        Name of the x-column in `data`.
    y_col : str
        Name of the y-column in `data`.
    content_col : str
        Name of the column in `data` with a description of the data point.
    x_grid: :class:`numpy.ndarray`
        The grid of x-values for which the y-values need to be calculated.

    Returns
    -------
    result : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        DataFrame with the points of the interpolated curve.
    """
    nw_data = nw.from_native(data, eager_only=True, pass_through=False)
    x_values = nw_data.get_column(x_col).to_numpy()
    y_values = nw_data.get_column(y_col).to_numpy()
    content_values = nw_data.get_column(content_col).to_numpy()

    content_col_0 = content_col + "0"
    content_col_1 = content_col + "1"

    interpolation_indices = _get_interpolation_indices(x_grid, x_values)

    x_distance_from_next_data_point = x_values[interpolation_indices + 1] - x_grid
    x_distance_between_data_points = (
        x_values[interpolation_indices + 1] - x_values[interpolation_indices]
    )
    p0 = x_distance_from_next_data_point / x_distance_between_data_points
    p1 = 1 - p0
    y = p0 * y_values[interpolation_indices] + p1 * y_values[interpolation_indices + 1]

    return nw.from_dict(
        {
            x_col: x_grid,
            y_col: y,
            "p0": p0,
            content_col_0: content_values[interpolation_indices],
            "p1": p1,
            content_col_1: content_values[interpolation_indices + 1],
        },
        backend=nw.get_native_namespace(data),
    ).to_native()


def _get_interpolation_indices(x_grid: NDArray, x_values: NDArray) -> NDArray:
    """
    This function finds the indices of `x_values` that are just to the left of each value in `x_grid`.
    The returned indices are such that:
    `x_values[indices[i]] < x_grid[i] <= x_values[indices[i] + 1]`.
    Parameters
    ----------
    x_grid : array-like
        The grid of x values where interpolation is desired.
    x_values : array-like
        The sorted array of x values from which to interpolate.
    Returns
    -------
    indices : ndarray
        The array of indices in `x_values` that are just to the left of each value in `x_grid`.
    """
    # Use 'right' to get the index of the first element in x_values greater than x_grid
    # Subtract 1 to get the index to the left
    # This yields indices that verify x_values[indices[i]] <= x_grid[i] < x_values[indices[i] + 1]
    # The edge case for the last element is handled by the second line.
    indices = np.searchsorted(x_values, x_grid, side="right") - 1

    # The convention is that in case a point in x_grid is equal to a point in x_values,
    # the index is the one to the left of the point in x_values.
    # i.e., x_values[indices[i]] < x_grid[i] <= x_values[indices[i] + 1]
    # This is achieved by decrementing the index if the point in x_values is equal to the point in x_grid.
    # This also keeps the last index in bounds.
    # The first element is left unchanged to keep the indices in bounds.
    indices[1:] = np.where(x_grid[1:] == x_values[indices[1:]], indices[1:] - 1, indices[1:])

    return indices


def _calculate_tradeoff_points(
    data: IntoDataFrameT,
    sensitive_feature_value,
    flip: bool = False,
    x_metric: Literal[
        "selection_rate",
        "true_positive_rate",
        "false_positive_rate",
        "true_negative_rate",
        "false_negative_rate",
    ] = "false_positive_rate",
    y_metric: Literal[
        "accuracy_score",
        "balanced_accuracy_score",
        "selection_rate",
        "true_positive_rate",
        "true_negative_rate",
    ] = "true_positive_rate",
) -> IntoDataFrameT:
    """Calculate the ROC points from the scores and labels.

    This is done by iterating through all possible
    thresholds that could be set based on the available scores.

    Parameters
    ----------
    data : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        The DataFrame containing scores and labels.
    sensitive_feature_value : str, int
        The sensitive feature value of the samples provided in `data`.
    flip : bool, default = True
        If True `flip` points below the ROC diagonal into points above by
        applying negative weights; if False does not allow flipping.
    x_metric : str, default="false_positive_rate"
        The metric to use for the x-axis. Possible values are:
        "selection_rate", "true_positive_rate", "false_positive_rate",
        "true_negative_rate", "false_negative_rate".
    y_metric : str, default="true_positive_rate"
        The metric to use for the y-axis. Possible values are:
        "accuracy_score", "balanced_accuracy_score", "selection_rate",
        "true_positive_rate", "true_negative_rate".

    Returns
    -------
    result : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        The ROC curve points with their corresponding threshold operations.
    """
    scores, labels, n, n_positive, n_negative = _get_scores_labels_and_counts(data)

    if n_positive == 0 or n_negative == 0:
        raise ValueError(DEGENERATE_LABELS_ERROR_MESSAGE.format(sensitive_feature_value))

    scores.append(-np.inf)
    labels.append(np.nan)

    # Iterate through all samples which are sorted by decreasing scores.
    # Setting the threshold between two scores means that everything smaller
    # than the threshold gets a label of 0 while everything larger than the
    # threshold gets a label of 1. Flipping labels is an option if flipping
    # labels provides better accuracy.
    i = 0
    count = [0, 0]
    x_list, y_list, operation_list = [], [], []

    x_metric_fn = METRIC_DICT[x_metric]
    y_metric_fn = METRIC_DICT[y_metric]

    while i < n:
        # special handling of the initial point
        if x_list == []:
            threshold = np.inf
        else:
            threshold = scores[i]
            while scores[i] == threshold:
                count[labels[i]] += 1
                i += 1
            threshold = (threshold + scores[i]) / 2

        actual_counts = _extend_confusion_matrix(
            false_positives=count[0],
            true_positives=count[1],
            true_negatives=(n_negative - count[0]),
            false_negatives=(n_positive - count[1]),
        )
        operations = [(">", actual_counts)]

        if flip:
            flipped_counts = _extend_confusion_matrix(
                false_positives=(n_negative - count[0]),
                true_positives=(n_positive - count[1]),
                true_negatives=count[0],
                false_negatives=count[1],
            )

            operations.append(("<", flipped_counts))

        for operation_string, counts in operations:
            x_list.append(x_metric_fn(counts))
            y_list.append(y_metric_fn(counts))
            operation_list.append(ThresholdOperation(operation_string, threshold))

    return nw.maybe_reset_index(
        nw.from_dict(
            {"x": x_list, "y": y_list, "operation": operation_list},
            backend=nw.get_native_namespace(data),
        ).sort(by=["x", "y"])
    ).to_native()


def _get_scores_labels_and_counts(data: IntoDataFrame) -> Tuple[list, list, int, int, int]:
    """Order samples by scores, counting number of positive, negative, and overall samples.

    The samples are sorted into descending order.

    Parameters
    ----------
    data : :class:`pandas:pandas.DataFrame`, :class:`polars:polars.DataFrame` or :class:`pyarrow:pyarrow.Table`
        The DataFrame containing scores and labels.

    Returns
    -------
    result : tuple[list, list, int, int, int]
        A tuple containing the sorted scores, labels, the number of samples, \
        the number of positive samples, and the number of negative samples.
    """
    data_sorted = nw.from_native(data, eager_only=True, pass_through=False).sort(
        by=SCORE_KEY, descending=True
    )

    scores = data_sorted.get_column(SCORE_KEY)
    labels = data_sorted.get_column(LABEL_KEY)

    n, n_positive, n_negative = _get_counts(labels)

    return scores.to_list(), labels.to_list(), n, n_positive, n_negative


def _get_counts(labels: nw.Series) -> Tuple[int, int, int]:
    """Return the overall, positive, and negative counts of the labels.

    Parameters
    ----------
    labels : list
        The labels of the samples.

    Returns
    -------
    result : tuple[int, int, int]
        A tuple containing the overall, positive, and negative counts of the labels.
    """
    n = len(labels)
    n_positive = labels.sum()
    n_negative = n - n_positive
    return n, n_positive, n_negative
