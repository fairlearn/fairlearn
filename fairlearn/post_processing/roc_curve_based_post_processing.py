# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors

tab10_norm  = matplotlib.colors.Normalize(vmin=0, vmax=7)
tab10_scalarMap = cm.ScalarMappable(norm=tab10_norm, cmap='Dark2')
debug_colors = [tab10_scalarMap.to_rgba(x) for x in range(10)]
debug_ncolors = len(debug_colors)
debug_colormap = {}

debug_markers = "^vso<>"
debug_nmarkers = len(debug_markers)
debug_markermap = {}

highlight_color = [0.95, 0.90, 0.40]

OUTPUT_SEPARATOR = "-"*65

def debug_marker(key):
    if key not in debug_markermap:
        marker = debug_markers[len(debug_markermap) % debug_nmarkers]
        debug_markermap[key] = marker
    return debug_markermap[key]

def debug_has_marker(key):
    return key in debug_markermap

def debug_color(key):
    if key not in debug_colormap:
        color = debug_colors[len(debug_colormap) % debug_ncolors]
        debug_colormap[key] = color
    return debug_colormap[key]

def debug_has_color(key):
    return key in debug_colormap

def pred_from_operator(op, threshold):
    """ Encodes the threshold rule P_hat > t or P_hat < t
    """
    if op == '>':
        return lambda x: x > threshold
    elif op == '<':
        return lambda x: x < threshold
    else:
        assert False, "Unrecognized operation:" + op

def interpolated_pred(p_ignore, pred_const, p0, op0, p1, op1):
    pred0 = pred_from_operator(*op0)
    pred1 = pred_from_operator(*op1)
    return (lambda x : p_ignore * pred_const + (1 - p_ignore) * (p0 * pred0(x) + p1 * pred1(x)))

def interpolate_curve(data, x_col, y_col, content_col, x_grid):
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
    for x in x_grid:
        while x > data_transpose[i + 1][x_col]:
            i += 1
        p0 = (data_transpose[i + 1][x_col] - x)/(data_transpose[i + 1][x_col] - data_transpose[i][x_col])
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

def get_roc(data, x_grid, flip=True, debug=False, attribute=None):
    """Get ROC curve based on data columns 'score' and 'label'"""

    attribute_str = "attribute value" + str(attribute)
    if debug:
        color = debug_color(attribute_str)
            
    data_sorted = data.sort_values(by='score', ascending=False)

    scores = list(data_sorted['score'])
    labels = list(data_sorted['label'])

    n = len(labels)
    n_positive = sum(labels)
    n_negative = n - n_positive

    assert (n_positive > 0) and (n_negative > 0), "Degenerate labels for " + attribute_str
    
    scores.append(-np.inf)
    labels.append(np.nan)
    
    x_list, y_list, op_list = [0], [0], [('>', np.inf)]
    
    i = 0
    count = [0, 0]
    while i < n:
        threshold = scores[i]
        while scores[i] == threshold:
            count[labels[i]] += 1
            i += 1
        x, y = count[0] / n_negative, count[1] / n_positive
        threshold = (threshold + scores[i]) / 2
        op = ('>', threshold)

        # if flipping labels gives better accuracy try flipping
        if flip & (x > y):
            x, y = 1 - x, 1 - y
            op = ('<', threshold)
        x_list.append(x)
        y_list.append(y)
        op_list.append(op)
    
    roc_raw = pd.DataFrame({'x': x_list, 'y': y_list, 'op': op_list})
    
    roc_sorted = roc_raw.sort_values(by=['x', 'y'])
    selected = []
    for r2 in roc_sorted.itertuples():
        while len(selected) >= 2:
            r1 = selected[-1]
            r0 = selected[-2]
            if (r1.y - r0.y) * (r2.x - r0.x) <= (r2.y - r0.y) * (r1.x - r0.x):
                selected.pop()
            else:
                break
        selected.append(r2)

    roc_conv = pd.DataFrame(selected)[['x', 'y', 'op']]

    roc_conv['sel'] = (n_negative / n) * roc_conv['x'] + (n_positive / n) * roc_conv['y']
    roc_conv['err'] = (n_negative / n) * roc_conv['x'] + (n_positive / n) * (1 - roc_conv['y'])
    
    roc_curve_interpolated = interpolate_curve(roc_conv, 'x', 'y', 'op', x_grid)
    sel_interp = interpolate_curve(roc_conv, 'sel', 'err', 'op', x_grid)

    if debug:
        print("")
        print(OUTPUT_SEPARATOR)
        print("")
        print(OUTPUT_SEPARATOR)
        print("Processing " + attribute_str)
        print(OUTPUT_SEPARATOR)
        print("DATA")
        print(data)
        print("\nROC curve: initial")
        #print(roc_raw)
        print(roc_sorted)
        print("\nROC curve: convex")
        print(roc_conv)
        print("\nROC curve: interpolated [just top]")
        print(roc_curve_interpolated.head())
        plt.plot(roc_sorted['x'], roc_sorted['y'], c=color, ls='--', lw=1.0, label='_')
        plt.plot(roc_conv['x'], roc_conv['y'], c=color, ls='-', lw=2.0, label='attribute ' + str(attribute))
        
    return roc_curve_interpolated, sel_interp
        
def roc_curve_based_post_processing(attributes, labels, scores, flip=True, debug=False, gridsize=1000):
    """ Post processing algorithm based on M. Hardt, E. Price, N. Srebro's paper "Equality of
    Opportunity in Supervised Learning" (https://arxiv.org/pdf/1610.02413.pdf).
    
    :param attributes: the protected attributes
    :type attributes: list
    :param labels: the labels of the dataset
    :type labels: list
    :param scores: the scores produced by a model's prediction
    :type scores: list
    :param flip: allow flipping to negative weights if it improves accuracy.
    :type flip: bool
    :param debug: show debugging output if True
    :type debug: bool
    :param gridsize: The number of ticks on the grid over which we evaluate the curves.
        A large gridsize means that we approximate the actual curve, so it increases the chance
        of being very close to the actual best solution.
    """
    data = pd.DataFrame({'attribute': attributes, 'score': scores, 'label': labels})
    assert len(labels) > 0, "Empty dataset"

    data_grouped_by_attribute = data.groupby('attribute')

    pred_EO = _roc_curve_based_post_processing_equalized_odds(labels, data_grouped_by_attribute, gridsize, flip, debug)
    pred_DP = _roc_curve_based_post_processing_demographic_parity(labels, data_grouped_by_attribute, gridsize, flip, debug)

    return pred_EO, pred_DP

def _roc_curve_based_post_processing_demographic_parity(labels, data_grouped_by_attribute, gridsize, flip, debug):
    n = len(labels)
    n_positive = sum(labels)
    n_negative = n - n_positive
    roc = {}
    sel = {}
    x_grid = np.linspace(0, 1, gridsize + 1)
    err_given_sel = 0 * x_grid
    for attribute, group in data_grouped_by_attribute:
        p_attribute = len(group) / n
        roc[attribute], sel[attribute] = get_roc(group, x_grid, flip=flip, debug=debug, attribute=attribute)
        err_given_sel += p_attribute * sel[attribute]['err']

    i_best_DP = err_given_sel.idxmin()
    sel_best = x_grid[i_best_DP]
    
    # create the solution as interpolation of multiple points with a separate predictor per protected attribute
    pred_DP_by_attribute = {}
    for attribute in roc.keys():
        # for DP we already have the predictor directly without complex interpolation. no p_ignore
        r = sel[attribute].transpose()[i_best_DP]
        pred_DP_by_attribute[attribute] = interpolated_pred(0, 0, r.p0, r.op0, r.p1, r.op1)
    
    if debug:
        print(OUTPUT_SEPARATOR)
        print("From ROC curves")
        print("Best DP: error=%.3f, selection rate=%.3f" % (err_given_sel[i_best_DP], sel_best))
        print(OUTPUT_SEPARATOR)

    return lambda a,x : pred_DP_by_attribute[a](x)   

def _roc_curve_based_post_processing_equalized_odds(labels, data_grouped_by_attribute, gridsize, flip, debug):
    n = len(labels)
    n_positive = sum(labels)
    n_negative = n - n_positive
    roc = {}
    sel = {}
    x_grid= np.linspace(0, 1, gridsize + 1)
    y_vals = pd.DataFrame()
    err_given_sel = 0 * x_grid
    for attribute, group in data_grouped_by_attribute:
        p_attribute = len(group) / n
        roc[attribute], sel[attribute] = get_roc(group, x_grid, flip=flip, debug=debug, attribute=attribute)
        y_vals[attribute] = roc[attribute]['y']

    y_min = np.amin(y_vals, axis=1)
    # conditional probabilities represented as x -> P[Y_hat=1 | Y=0]
    # and                                      y -> P[Y_hat=1 | Y=1]
    err_given_x = (n_negative / n) * x_grid + (n_positive / n) * (1 - y_min)
    i_best_EO = err_given_x.idxmin()
    x_best = x_grid[i_best_EO]
    y_best = y_min[i_best_EO]
    
    # create the solution as interpolation of multiple points with a separate predictor per protected attribute
    pred_EO_by_attribute = {}
    for attribute in roc.keys():
        r = roc[attribute].transpose()[i_best_EO]
        # p_ignore is the probability at which we're ignoring the score, i.e. on the diagonal of the ROC curve
        if r.y == r.x:
            p_ignore = 0
        else:
            p_ignore = (r.y - y_best) / (r.y - r.x)
        pred_EO_by_attribute[attribute] = interpolated_pred(p_ignore, x_best, r.p0, r.op0, r.p1, r.op1)

    if debug:
        print(OUTPUT_SEPARATOR)
        print("From ROC curves")
        print("Best EO: error=%.3f, FP rate=%.3f, TP rate=%.3f" % (err_given_x[i_best_EO], x_best, y_best))
        print(OUTPUT_SEPARATOR)
        line, = plt.plot(x_grid, y_min, color=highlight_color, lw=8, label='overlap')
        line.zorder -= 1
        plt.plot(x_best, y_best, 'm*', ms=10, label='EO solution') 
        plt.legend()

    return lambda a,x : pred_EO_by_attribute[a](x)
