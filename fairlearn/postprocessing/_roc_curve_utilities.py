# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from ._constants import LABEL_KEY, SCORE_KEY, P0_KEY, P1_KEY
from ._threshold_operation import ThresholdOperation

DEGENERATE_LABELS_ERROR_MESSAGE = "Degenerate labels for sensitive feature value {}"


def _get_roc(data, sensitive_feature_value, flip=True):
    """Get ROC curve's convex hull based on data columns 'score' and 'label'.

    Scores represent output values from the predictor.

    :param data: the DataFrame containing scores and labels
    :type data: pandas.DataFrame
    :param sensitive_feature_value: the sensitive feature value of the samples provided in `data`
    :type sensitive_feature_value: str or int
    :param flip: if True flip points below the ROC diagonal into points above by applying negative
        weights; if False does not allow flipping; default True
    :type flip: bool
    :return: the convex hull over the ROC curve points
    :rtype: pandas.DataFrame
    """
    roc_sorted = _calculate_roc_points(data, sensitive_feature_value, flip)
    roc_selected = _filter_points_to_get_convex_hull(roc_sorted)
    roc_convex_hull = pd.DataFrame(roc_selected)[['x', 'y', 'operation']]
    return roc_convex_hull


def _filter_points_to_get_convex_hull(roc_sorted):
    """Finds the convex hull.

    Uses a simplified version of Andrew's monotone chain convex hull algorithm
    https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
    to get the convex hull. Since we can assume the points (0,0) and (1,1) to be part
    of the convex hull the problem is simpler and we only need to make a single pass
    through the data.

    :param roc_sorted: DataFrame with ROC curve points sorted by 'x'
    :type roc_sorted: pandas.DataFrame
    :return: the list of points that make up the convex hull
    :rtype: list of named tuples
    """
    selected = []
    for r2 in roc_sorted.itertuples():
        # For each set of three points, i.e. the last two points in selected
        # and the next point from the sorted list of base points, check
        # whether the middle point (r1) lies above the line between the
        # first point (r0) and the next base point (r2). If it is above,
        # it is indeed required for the convex hull. If it is below or
        # on the line, then it is part of the convex hull as defined with
        # just r0 and r2 and we can drop it from the list of selected points.
        while len(selected) >= 2:
            r1 = selected[-1]
            r0 = selected[-2]
            # Compare slopes of lines between r0 and r1/r2 to determine
            # whether or not to drop r1. Instead of delta_y/delta_x we
            # multiplied both sides of the inequation by the delta_xs.
            if (r1.y - r0.y) * (r2.x - r0.x) <= (r2.y - r0.y) * (r1.x - r0.x):
                # drop r1
                selected.pop()
            else:
                break
        selected.append(r2)
    return selected


def _interpolate_curve(data, x_col, y_col, content_col, x_grid):
    """Interpolates the DataFrame in `data` along the values in `x_grid`.

    Assumes: (1) data[y_col] is convex and non-decreasing in data[x_col]
             (2) min and max in x_grid are below/above min and max in data[x_col]
             (3) data is indexed 0,...,len(data)

    :param data: the convex hull data points
    :type data: pandas.DataFrame
    :param x_col: name of the x-column in `data`
    :type x_col: str
    :param y_col: name of the y-column in `data`
    :type y_col: str
    :param content_col: name of the column in `data` with a description of the data point
    :type content_col: str
    :param x_grid: the grid of x-values for which the y-values need to be calculated
    :type x_grid: numpy.ndarray
    :return: DataFrame with the points of the interpolated curve
    :type: pandas.DataFrame
    """
    data_transpose = data.transpose()

    content_col_0 = content_col + '0'
    content_col_1 = content_col + '1'

    i = 0
    dict_list = []
    x0 = data_transpose[0][x_col]
    while data_transpose[i + 1][x_col] == x0:
        i += 1

    # calculate the curve points for each x tick in x_grid
    for x in x_grid:
        # skip over data points that we've already passed
        while x > data_transpose[i + 1][x_col]:
            i += 1

        # Calculate the y value at x based on the slope between data points i and i + 1
        x_distance_from_next_data_point = data_transpose[i + 1][x_col] - x
        x_distance_between_data_points = data_transpose[i + 1][x_col] - data_transpose[i][x_col]
        p0 = x_distance_from_next_data_point/x_distance_between_data_points
        p1 = 1 - p0
        y = p0 * data_transpose[i][y_col] + p1 * data_transpose[i + 1][y_col]
        dict_list.append({
            x_col: x,
            y_col: y,
            P0_KEY: p0,
            content_col_0: data_transpose[i][content_col],
            P1_KEY: p1,
            content_col_1: data_transpose[i + 1][content_col]})

    return pd.DataFrame(dict_list)[[x_col, y_col, P0_KEY, content_col_0, P1_KEY, content_col_1]]


def _calculate_roc_points(data, sensitive_feature_value, flip=True):
    """Calculates the ROC points from the scores and labels.

    This is done by iterating through all possible
    thresholds that could be set based on the available scores.

    :param data: the DataFrame containing scores and labels
    :type data: pandas.DataFrame
    :param sensitive_feature_value: the sensitive feature value of the samples provided in `data`
    :type sensitive_feature_value: str or int
    :param flip: if True flip points below the ROC diagonal into points above by applying negative
        weights; if False does not allow flipping; default True
    :type flip: bool
    :return: the ROC curve points with their corresponding threshold operations
    :rtype: pandas.DataFrame
    """
    scores, labels, n, n_positive, n_negative = _get_scores_labels_and_counts(data)

    if n_positive == 0 or n_negative == 0:
        raise ValueError(DEGENERATE_LABELS_ERROR_MESSAGE.format(sensitive_feature_value))

    scores.append(-np.inf)
    labels.append(np.nan)

    x_list, y_list, operation_list = [0], [0], [ThresholdOperation('>', np.inf)]

    # Iterate through all samples which are sorted by increasing scores.
    # Setting the threshold between two scores means that everything smaller
    # than the threshold gets a label of 0 while everything larger than the
    # threshold gets a label of 1. Flipping labels is an option if flipping
    # labels provides better accuracy.
    i = 0
    count = [0, 0]
    while i < n:
        threshold = scores[i]
        while scores[i] == threshold:
            count[labels[i]] += 1
            i += 1
        # For the ROC curve we calculate points (x, y), where x represents
        # the conditional probability P[Y_hat=1 | Y=0] and y represents
        # the conditional probability P[Y_hat=1 | Y=1]. The conditional
        # probability is achieved by dividing by only the number of
        # negative/positive samples.
        x, y = count[0] / n_negative, count[1] / n_positive
        threshold = (threshold + scores[i]) / 2
        operation = ThresholdOperation('>', threshold)

        if flip and x > y:
            x, y = 1 - x, 1 - y
            operation = ThresholdOperation('<', threshold)
        x_list.append(x)
        y_list.append(y)
        operation_list.append(operation)

    return pd.DataFrame({'x': x_list, 'y': y_list, 'operation': operation_list}) \
        .sort_values(by=['x', 'y'])


def _get_scores_labels_and_counts(data):
    """Order samples by scores, counting number of positive, negative, and overall samples.

    The samples are sorted into ascending order.

    :param data: the DataFrame containing scores and labels
    :type data: pandas.DataFrame
    :return: a tuple containing the sorted scores, labels, the number of samples, the number
        of positive samples, and the number of negative samples
    :rtype: tuple of list, list, int, int, int
    """
    data_sorted = data.sort_values(by=SCORE_KEY, ascending=False)

    scores = list(data_sorted[SCORE_KEY])
    labels = list(data_sorted[LABEL_KEY])

    n, n_positive, n_negative = _get_counts(labels)

    return scores, labels, n, n_positive, n_negative


def _get_counts(labels):
    """Returns the overall, positive, and negative counts of the labels.

    :param labels: the labels of the samples
    :type labels: list
    :return: a tuple containing the overall, positive, and negative counts of the labels
    :rtype: tuple of int, int, int
    """
    n = len(labels)
    n_positive = sum(labels)
    n_negative = n - n_positive
    return n, n_positive, n_negative
