# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" ROC Curve based Post processing algorithm based on M. Hardt,
E. Price, N. Srebro's paper "Equality of Opportunity in Supervised
Learning" (https://arxiv.org/pdf/1610.02413.pdf) for binary
classification with one categorical protected attribute. 
"""

import sys
import numpy as np
import pandas as pd


from .roc_curve_plotting_utilities import plot_solution_and_show_plot, plot_overlap, plot

OUTPUT_SEPARATOR = "-"*65

SCORE_KEY = "score"
LABEL_KEY = "label"
ATTRIBUTE_KEY = "attribute"

DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE = "Attributes, labels, and scores need to be of equal length."
EMPTY_INPUT_ERROR_MESSAGE = "At least one of attributes, labels, or scores are empty."

class ThresholdOperation():
    def __init__(self, operator, threshold):
        if operator not in ['>', '<']:
            raise ValueError("Unrecognized operator: " + operator)
        self._operator = operator
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @property
    def operator(self):
        return self._operator

    def get_predictor_from_operation(self):
        """ Encodes the threshold rule Y_hat > t or Y_hat < t"""
        if self._operator == '>':
            return lambda x: x > self._threshold
        elif self._operator == '<':
            return lambda x: x < self._threshold
        else:
            raise ValueError("Unrecognized operator: " + self._operator)
    
    def __repr__(self):
        return "[{}{}]".format(self._operator, self._threshold)

def _interpolate_prediction(p_ignore, prediction_constant, p0, operation0, p1, operation1, debug=False):
    """Creates the interpolated prediction between two predictions. The predictions
    are represented through the threshold rules operation0 and operation1.
    
    :param p_ignore: p_ignore changes the interpolated prediction P to the desired 
        solution using the transformation p_ignore * prediction_constant + (1 - p_ignore) * P 
    :param prediction_constant: 0 if not required, otherwise the x value of the best
        solution should be passed
    :param p0: interpolation multiplier for prediction from the first predictor
    :param operation0: threshold rule for the first predictor
    :param p1: interpolation multiplier for prediction from the second predictor
    :param operation1: threshold rule for the second predictor
    :return: an anonymous function that scales the original prediction to the desired one
    """
    pred0 = operation0.get_predictor_from_operation()
    pred1 = operation1.get_predictor_from_operation()
    if debug:
        print(OUTPUT_SEPARATOR)
        print("p_ignore: {}".format(p_ignore))
        print("prediction_constant: {}".format(prediction_constant))
        print("p0: {}".format(p0))
        print("operation0: {}".format(operation0))
        print("p1: {}".format(p1))
        print("operation1: {}".format(operation1))
        print(OUTPUT_SEPARATOR)
    return (lambda x : p_ignore * prediction_constant + (1 - p_ignore) * (p0 * pred0(x) + p1 * pred1(x)))

def _interpolate_curve(data, x_col, y_col, content_col, x_grid):
    """Interpolates the data frame in "data" along the values in "x_grid".
    Assumes: (1) data[y_col] is convex and non-decreasing in data[x_col]
             (2) min and max in x_grid are below/above min and max in data[x_col]
             (3) data is indexed 0,...,len(data)"""
    data_transpose = data.transpose()

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
            'p0': p0,
            content_col + '0': data_transpose[i][content_col],
            'p1': p1,
            content_col + '1': data_transpose[i + 1][content_col]})

    return pd.DataFrame(dict_list)[[x_col, y_col, 'p0', content_col + '0', 'p1', content_col + '1']]

def _calculate_roc_points(data, attribute, flip=True):
    scores, labels, n, n_positive, n_negative = _get_scores_labels_and_counts(data)

    if n_positive == 0 or n_negative == 0:
        raise ValueError("Degenerate labels for attribute value {}".format(attribute))

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
        # the conditional probability P[Y_hat=0 | Y=0] and y represents
        # the conditional probability P[Y_hat=0 | Y=1]. The conditional
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
        
    return pd.DataFrame({'x': x_list, 'y': y_list, 'operation': operation_list}).sort_values(by=['x', 'y'])

def _get_scores_labels_and_counts(data):
    data_sorted = data.sort_values(by=SCORE_KEY, ascending=False)

    scores = list(data_sorted[SCORE_KEY])
    labels = list(data_sorted[LABEL_KEY])

    n, n_positive, n_negative = _get_counts(labels)

    return scores, labels, n, n_positive, n_negative

def _get_counts(labels):
    n = len(labels)
    n_positive = sum(labels)
    n_negative = n - n_positive
    return n, n_positive, n_negative

def _get_roc(data, x_grid, attribute, flip=True, debug=False):
    """Get ROC curve's convex hull based on data columns 'score' and 'label'
    Scores represent output values from the model.
    """
    roc_sorted = _calculate_roc_points(data, attribute, flip)
    roc_selected = _filter_points_to_get_convex_hull(roc_sorted)
    roc_convex_hull = pd.DataFrame(roc_selected)[['x', 'y', 'operation']]
    return roc_convex_hull

def _filter_points_to_get_convex_hull(roc_sorted):
    selected = []
    for r2 in roc_sorted.itertuples():
        while len(selected) >= 2:
            r1 = selected[-1]
            r0 = selected[-2]
            # Calculate the y value at r2's x if the slope between r0 and r1 is continued.
            # If that y value is not larger than r2's actual y value we know that r1 lies
            # below the line between r0 and r2 and can be dropped since we can reach all points
            # between r0 and r2 through interpolation.
            if (r1.y - r0.y) * (r2.x - r0.x) <= (r2.y - r0.y) * (r1.x - r0.x):
                selected.pop()
            else:
                break
        selected.append(r2)
    return selected
        
def _sanity_check_and_group_data(attributes, labels, scores):
    if len(attributes) == 0 or len(labels) == 0 or len(scores) == 0:
        raise ValueError(EMPTY_INPUT_ERROR_MESSAGE)

    if len(attributes) != len(labels) or len(attributes) != len(scores):
        raise ValueError(DIFFERENT_INPUT_LENGTH_ERROR_MESSAGE)
    data = pd.DataFrame({ATTRIBUTE_KEY: attributes, SCORE_KEY: scores, LABEL_KEY: labels})

    return data.groupby(ATTRIBUTE_KEY)
    

def roc_curve_based_post_processing_demographic_parity(attributes, labels, scores, gridsize=1000, flip=True, debug=False):
    """ TODO add description
    :param attributes: the protected attributes
    :type attributes: list
    :param labels: the labels of the dataset
    :type labels: list
    :param scores: the scores produced by a model's prediction
    :type scores: list
    :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    :type gridsize: int
    :param flip: allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param debug: show debugging output if True
    :type debug: bool
    """
    n = len(labels)
    selection_error_curve = {}
    x_grid = np.linspace(0, 1, gridsize + 1)
    error_given_selection = 0 * x_grid

    data_grouped_by_attribute = _sanity_check_and_group_data(attributes, labels, scores)

    for attribute, group in data_grouped_by_attribute:
        # determine probability of current protected attribute group based on data
        n_group = len(group)
        n_positive = sum(group[LABEL_KEY])
        n_negative = n_group - n_positive
        p_attribute = n_group / n

        roc_convex_hull = _get_roc(group, x_grid, attribute, flip=flip, debug=debug)

        fraction_negative_label_positive_sample = (n_negative / n_group) * roc_convex_hull['x']
        fraction_positive_label_positive_sample = (n_positive / n_group) * roc_convex_hull['y']
        # Calculate selection to represent the proportion of positive predictions.
        roc_convex_hull['selection'] = fraction_negative_label_positive_sample + fraction_positive_label_positive_sample

        fraction_positive_label_negative_sample = (n_positive / n_group) * (1 - roc_convex_hull['y'])
        roc_convex_hull['error'] = fraction_negative_label_positive_sample + fraction_positive_label_negative_sample

        selection_error_curve[attribute] = _interpolate_curve(roc_convex_hull, 'selection', 'error', 'operation', x_grid)
        
        # Add up errors for the current group multiplied by the probability of the current group.
        # This will help us in identifying the minimum overall error.
        error_given_selection += p_attribute * selection_error_curve[attribute]['error']

        if debug:
            print("")
            print(OUTPUT_SEPARATOR)
            print("Processing " + attribute)
            print(OUTPUT_SEPARATOR)
            print("DATA")
            print(group)
            print("\nROC curve: convex")
            print(roc_convex_hull)
            plot(attribute, 'selection', 'error', selection_error_curve[attribute])

    # Find minimum error point given that at each point the selection rate for each attribute
    # value is identical by design.
    i_best_DP = error_given_selection.idxmin()
    x_best = x_grid[i_best_DP]
    
    # create the solution as interpolation of multiple points with a separate predictor per protected attribute
    predicted_DP_by_attribute = {}
    for attribute in selection_error_curve.keys():
        # For DP we already have the predictor directly without complex interpolation.
        roc_result = selection_error_curve[attribute].transpose()[i_best_DP]
        predicted_DP_by_attribute[attribute] = _interpolate_prediction(0, 0,
                                                                       roc_result.p0, roc_result.operation0,
                                                                       roc_result.p1, roc_result.operation1,
                                                                       debug)
    
    if debug:
        print(OUTPUT_SEPARATOR)
        print("From ROC curves")
        print("Best DP: error=%.3f, selection rate=%.3f" % (error_given_selection[i_best_DP], x_best))
        print(OUTPUT_SEPARATOR)
        plot_solution_and_show_plot(x_best, None, "DP solution")

    return lambda a, x: predicted_DP_by_attribute[a](x)   

def roc_curve_based_post_processing_equalized_odds(attributes, labels, scores, gridsize=1000, flip=True, debug=False):
    """ Calculates the ROC curve of every attribute and take the overlapping region.
    From the resulting ROC curve the algorithm finds the best solution by selecting the
    point on the curve with minimal error.

    :param attributes: the protected attributes
    :type attributes: list
    :param labels: the labels of the dataset
    :type labels: list
    :param scores: the scores produced by a model's prediction
    :type scores: list
    :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    :type gridsize: int
    :param flip: allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param debug: show debugging output if True
    :type debug: bool
    """
    n = len(labels)
    n_positive = sum(labels)
    n_negative = n - n_positive
    roc = {}
    x_grid = np.linspace(0, 1, gridsize + 1)
    y_values = pd.DataFrame()

    data_grouped_by_attribute = _sanity_check_and_group_data(attributes, labels, scores)

    for attribute, group in data_grouped_by_attribute:
        roc_convex_hull = _get_roc(group, x_grid, attribute, flip=flip, debug=debug)
        roc[attribute] = _interpolate_curve(roc_convex_hull, 'x', 'y', 'operation', x_grid)
        y_values[attribute] = roc[attribute]['y']

        if debug:
            print("")
            print(OUTPUT_SEPARATOR)
            print("Processing " + attribute)
            print(OUTPUT_SEPARATOR)
            print("DATA")
            print(group)
            print("\nROC curve: convex")
            print(roc_convex_hull)
            plot(attribute, 'x', 'y', roc[attribute])

    # Calculate the overlap of the ROC curves by taking the lowest y value
    # at every given x.
    y_min = np.amin(y_values, axis=1)
    # Calculate the error at any given x as the sum of
    # a) the proportion of negative labels multiplied by x which represents
    #    the conditional probability P[Y_hat=1 | Y=0], i.e. the probability
    #    of a positive prediction given a negative label.
    # b) the propotion of positive labels multiplied by 1-y_min, where y_min
    #    represents the conditional probability P[Y_hat=1 | Y=1], i.e. the
    #    probability of a correct prediction of a positive label, so 1-y_min
    #    represents a negative prediction given a positive label.
    error_given_x = (n_negative / n) * x_grid + (n_positive / n) * (1 - y_min)
    i_best_EO = error_given_x.idxmin()
    x_best = x_grid[i_best_EO]
    y_best = y_min[i_best_EO]
    
    # create the solution as interpolation of multiple points with a separate predictor per protected attribute
    predicted_EO_by_attribute = {}
    for attribute in roc.keys():
        roc_result = roc[attribute].transpose()[i_best_EO]
        # p_ignore * x_best represent the diagonal of the ROC plot.
        if roc_result.y == roc_result.x:
            # result is on the diagonal of the ROC plot, i.e. p_ignore is not required
            p_ignore = 0
        else:
            # Calculate p_ignore to change prediction P to y_best
            # p_ignore * x_best + (1 - p_ignore) * P
            difference_from_best_predictor_for_attribute = roc_result.y - y_best
            vertical_distance_from_diagonal = roc_result.y - roc_result.x
            p_ignore = difference_from_best_predictor_for_attribute / vertical_distance_from_diagonal

        predicted_EO_by_attribute[attribute] = _interpolate_prediction(p_ignore, x_best,
                                                                       roc_result.p0, roc_result.operation0,
                                                                       roc_result.p1, roc_result.operation1,
                                                                       debug)

    if debug:
        print(OUTPUT_SEPARATOR)
        print("From ROC curves")
        print("Best EO: error=%.3f, FP rate=%.3f, TP rate=%.3f" % (error_given_x[i_best_EO], x_best, y_best))
        print(OUTPUT_SEPARATOR)
        plot_overlap(x_grid, y_min)
        plot_solution_and_show_plot(x_best, y_best, 'EO solution')
        
    return lambda a, x: predicted_EO_by_attribute[a](x)
