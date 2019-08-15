# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from fairlearn.post_processing.roc_curve_based_post_processing import roc_curve_based_post_processing

ex_attrs1 = [x      for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
ex_attrs2 = [x      for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']
ex_labels = [int(x) for x in '0110100' '0010111' '000111']
ex_scores = [int(x) for x in '0011233' '0001111' '011112']


def run_roc_curve_based_post_processing_and_plot(ex_attrs, flip):
    print("STARTING TEST")
    pred_EO, pred_DP = roc_curve_based_post_processing(ex_attrs, ex_labels, ex_scores, debug=True, flip=flip)
    ex_preds_EO = []
    ex_preds_DP = []
    for i in range(len(ex_attrs)):
        ex_preds_EO.append( pred_EO(ex_attrs[i], ex_scores[i]) )
        ex_preds_DP.append( pred_DP(ex_attrs[i], ex_scores[i]) )
    ex_data = pd.DataFrame({'attr': ex_attrs, 'score': ex_scores, 'label': ex_labels, 'pred_EO': ex_preds_EO, 'pred_DP': ex_preds_DP})
    ex_data['error_EO'] = np.absolute(ex_data['label']-ex_data['pred_EO'])
    ex_data['error_DP'] = np.absolute(ex_data['label']-ex_data['pred_DP'])
    #print("DATA")
    #print( ex_data[['attr','label','pred']] )
    print("APPLYING EO PREDICTOR")
    print("")
    print(ex_data.groupby( ['attr','label'] ).mean()[['pred_EO']] )
    print("")
    print("error_EO=%.3f" % ex_data['error_EO'].mean() )

    print("-"*65)

    print("APPLYING DP PREDICTOR")
    print("")
    print(ex_data.groupby( ['attr'] ).mean()[['pred_DP']] )
    print("")
    print("error_DP=%.3f" % ex_data['error_DP'].mean() )
    plt.show()

def test_1():
    run_roc_curve_based_post_processing_and_plot(ex_attrs1, True)

def test_2():
    run_roc_curve_based_post_processing_and_plot(list(zip(ex_attrs1, ex_attrs2)), True)

def test_3():
    run_roc_curve_based_post_processing_and_plot(ex_attrs1, False)

def test_4():
    run_roc_curve_based_post_processing_and_plot(ex_attrs2, False)
