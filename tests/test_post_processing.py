# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.post_processing.roc_curve_based_post_processing import equalized_odds

ex_attrs1 = [x      for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
ex_attrs2 = [x      for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']
ex_labels = [int(x) for x in '0110100' '0010111' '000111']
ex_scores = [int(x) for x in '0011233' '0001111' '011112']


def test(test_id=1):
    print("STARTING TEST")
    if test_id==1:
        ex_attrs, flip = ex_attrs1, True
    elif test_id==2:
        ex_attrs, flip = list(zip(ex_attrs1, ex_attrs2)), True
    elif test_id==3:
        ex_attrs, flip = ex_attrs1, False
    elif test_id==4:
        ex_attrs, flip = ex_attrs2, False
    
    pred_EO, pred_DP = equalized_odds(ex_attrs, ex_labels, ex_scores, debug=True, flip=flip)
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

if __name__ == '__main__':
    test(1)
    test(2)
    test(3)
    test(4)