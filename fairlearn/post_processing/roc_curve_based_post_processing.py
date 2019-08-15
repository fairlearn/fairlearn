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

def pred_from_op(op, th):
    """ encodes the threshold rule P_hat > t or P_hat < t
    """
    if op=='>':
        return lambda x: x>th
    elif op=='<':
        return lambda x: x<th
    else:
        assert False, "Unrecognized operation:" +op

def interpolated_pred(p_ignore, pred_const, p0, op0, p1, op1):
    pred0 = pred_from_op(*op0)
    pred1 = pred_from_op(*op1)
    return (lambda x :
        p_ignore*pred_const + (1-p_ignore)*(p0*pred0(x) + p1*pred1(x))
        )

def interpolate_curve(data, x_col, y_col, content_col, x_grid):
    """Interpolates the data frame in "data" along the values in "x_grid".

    Assumes: (1) data[y_col] is convex and non-decreasing in data[x_col]
             (2) min and max in x_grid are below/above min and max in data[x_col]
             (3) data is indexed 0,...,len(data)"""

    data_tr = data.transpose()

    i = 0
    dict_list = []
    x0 = data_tr[0][x_col]
    while data_tr[i+1][x_col] == x0:
        i += 1
    for x in x_grid:
        while x>data_tr[i+1][x_col]:
            i += 1
        p0 = (data_tr[i+1][x_col]-x)/(data_tr[i+1][x_col]-data_tr[i][x_col])
        p1 = 1-p0
        y = p0*data_tr[i][y_col] + p1*data_tr[i+1][y_col]
        dict_list.append(
            {x_col: x, y_col: y,
             'p0': p0, content_col+'0': data_tr[i  ][content_col],
             'p1': p1, content_col+'1': data_tr[i+1][content_col]})

    return pd.DataFrame( dict_list )[[x_col, y_col, 'p0', content_col+'0', 'p1', content_col+'1']]

def get_roc(data, x_grid, flip=True, debug=False, attr=None):
    """Get ROC curve based on data columns 'score' and 'label'"""

    attr_str = "attribute "+str(attr)
    if debug:
        color = debug_color(attr_str)
            
    data_sorted = data.sort_values(by='score', ascending=False)

    scores = list(data_sorted['score'])
    labels = list(data_sorted['label'])

    n = len(labels)
    npos = sum(labels)
    nneg = n-npos

    assert (npos>0) & (nneg>0), "Degenerate labels for "+attr_str
    
    scores.append(-np.inf)
    labels.append( np.nan)
    
    x_list, y_list, op_list = [0], [0], [('>',np.inf)]
    
    i = 0
    count = [0, 0]
    while i<n:
        th = scores[i]
        while scores[i]==th:
            count[labels[i]] += 1
            i += 1
        x, y = count[0]/nneg, count[1]/npos
        th = (th+scores[i])/2
        op = ('>', th)

        # if flipping labels gives better accuracy try flipping
        if flip & (x > y):
            x, y = 1-x, 1-y
            op = ('<', th)
        x_list.append(x)
        y_list.append(y)
        op_list.append(op)
    
    roc_raw = pd.DataFrame({'x': x_list, 'y': y_list, 'op': op_list})
    
    roc_sorted = roc_raw.sort_values(by=['x', 'y'])
    selected = []
    for r2 in roc_sorted.itertuples():
        while len(selected)>=2:
            r1 = selected[-1]
            r0 = selected[-2]
            if (r1.y-r0.y)*(r2.x-r0.x) <= (r2.y-r0.y)*(r1.x-r0.x):
                selected.pop()
            else:
                break
        selected.append(r2)

    roc_conv = pd.DataFrame(selected)[['x','y','op']]

    roc_conv['sel'] = (nneg/n)*roc_conv['x'] + (npos/n)*roc_conv['y']
    roc_conv['err'] = (nneg/n)*roc_conv['x'] + (npos/n)*(1-roc_conv['y'])
    
    roc_interp = interpolate_curve(roc_conv, 'x', 'y', 'op', x_grid)
    sel_interp = interpolate_curve(roc_conv, 'sel', 'err', 'op', x_grid)

    if debug:
        print("")
        print("#"*65)
        print("")
        print("-"*65)
        print("Processing "+attr_str)
        print("-"*65)
        print("DATA")
        print(data)
        print("\nROC curve: initial")
        #print(roc_raw)
        print(roc_sorted)
        print("\nROC curve: convex")
        print(roc_conv)
        print("\nROC curve: interpolated [just top]")
        print(roc_interp.head())
        plt.plot(roc_sorted['x'], roc_sorted['y'], c=color, ls='--', lw=1.0, label='_')
        plt.plot(roc_conv['x'], roc_conv['y'], c=color, ls='-', lw=2.0, label='attribute '+str(attr))
        
    return roc_interp, sel_interp
        
def roc_curve_based_post_processing(attrs, labels, scores, flip=True, debug=False, gridsize=1000):
    """ Post processing algorithm based on M. Hardt, E. Price, N. Srebro's paper "Equality of
    Opportunity in Supervised Learning" (https://arxiv.org/pdf/1610.02413.pdf).
    
    :param attrs: the protected attributes
    :type attrs: list
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
    data = pd.DataFrame({'attr': attrs, 'score': scores, 'label': labels})
    assert len(labels) > 0, "Empty dataset"

    grouped = data.groupby('attr')

    pred_EO = _roc_curve_based_post_processing_equalized_odds(labels, grouped, gridsize, flip, debug)
    pred_DP = _roc_curve_based_post_processing_demographic_parity(labels, grouped, gridsize, flip, debug)

    return pred_EO, pred_DP

def _roc_curve_based_post_processing_demographic_parity(labels, grouped, gridsize, flip, debug):
    n = len(labels)
    npos = sum(labels)
    nneg = n-npos
    roc = {}
    sel = {}
    x_grid= np.linspace(0, 1, gridsize+1)
    err_given_sel = 0*x_grid
    for attr, group in grouped:
        p_attr = len(group)/n
        roc[attr], sel[attr] = get_roc(group, x_grid, flip=flip, debug=debug, attr=attr)
        err_given_sel += p_attr * sel[attr]['err']

    i_best_DP = err_given_sel.idxmin()
    sel_best = x_grid[i_best_DP]
    
    # create the solution as interpolation of multiple points with a separate predictor per protected attribute
    pred_DP_by_attr = {}
    for attr in roc.keys():
        # for DP we already have the predictor directly without complex interpolation. no p_ignore
        r = sel[attr].transpose()[i_best_DP]
        pred_DP_by_attr[attr] = interpolated_pred(0, 0, r.p0, r.op0, r.p1, r.op1)
    
    if debug:
        print("-"*65)
        print("From ROC curves")
        print("Best DP: error=%.3f, selection rate=%.3f" % (err_given_sel[i_best_DP], sel_best))
        print("-"*65)

    return lambda a,x : pred_DP_by_attr[a](x)   

def _roc_curve_based_post_processing_equalized_odds(labels, grouped, gridsize, flip, debug):
    n = len(labels)
    npos = sum(labels)
    nneg = n-npos
    roc = {}
    sel = {}
    x_grid= np.linspace(0, 1, gridsize+1)
    y_vals = pd.DataFrame()
    err_given_sel = 0*x_grid
    for attr, group in grouped:
        p_attr = len(group)/n
        roc[attr], sel[attr] = get_roc(group, x_grid, flip=flip, debug=debug, attr=attr)
        y_vals[attr] = roc[attr]['y']

    # EQUALIZED ODDS
    y_min = np.amin(y_vals, axis=1)
    # conditional probabilities represented as x -> P[Y_hat=1 | Y=0]
    # and                                      y -> P[Y_hat=1 | Y=1]
    err_given_x = (nneg/n) * x_grid + (npos/n) * (1-y_min)
    i_best_EO = err_given_x.idxmin()
    x_best = x_grid[i_best_EO]
    y_best = y_min[i_best_EO]
    
    # create the solution as interpolation of multiple points with a separate predictor per protected attribute
    pred_EO_by_attr = {}
    for attr in roc.keys():
        r = roc[attr].transpose()[i_best_EO]
        # p_ignore is the probability at which we're ignoring the score, i.e. on the diagonal of the ROC curve
        if r.y == r.x:
            p_ignore = 0
        else:
            p_ignore = (r.y - y_best) / (r.y - r.x)
        pred_EO_by_attr[attr] = interpolated_pred(p_ignore, x_best, r.p0, r.op0, r.p1, r.op1)

    if debug:
        print("-"*65)
        print("From ROC curves")
        print("Best EO: error=%.3f, FP rate=%.3f, TP rate=%.3f" % (err_given_x[i_best_EO], x_best, y_best))
        print("-"*65)
        line, = plt.plot(x_grid, y_min, color=highlight_color, lw=8, label='overlap')
        line.zorder -= 1
        plt.plot(x_best, y_best, 'm*', ms=10, label='EO solution') 
        plt.legend()

    return lambda a,x : pred_EO_by_attr[a](x)
