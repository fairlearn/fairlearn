# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from sklearn.utils import Bunch

SCORE_KEY = "score"
LABEL_KEY = "label"
SENSITIVE_FEATURE_KEY = "sensitive_feature"
P0_KEY = "p0"
P1_KEY = "p1"

OUTPUT_SEPARATOR = "-"*65

DEMOGRAPHIC_PARITY = "demographic_parity"
EQUALIZED_ODDS = "equalized_odds"
ACCURACY_SCORE = "accuracy_score"

def confusion_matrix_summary(*, true_positives, false_positives, true_negatives, false_negatives):
    """Return the Bunch with the fields required for confusion-matrix metrics."""
    
    return Bunch(
        true_positives = true_positives,
        false_positives = false_positives,
        true_negatives = true_negatives,
        false_negatives = false_negatives,
        predicted_positives = true_positives + false_positives,
        predicted_negatives = true_negatives + false_negatives,
        positives = true_positives + false_negatives,
        negatives = true_negatives + false_positives,
        n = true_positives + true_negatives + false_positives + false_negatives,
    )

# Dictionary of metrics based on confusion matrix. Their input must be a Bunch with the fields
# named n, positives, negatives, predicted_positives, predicted_negatives, true_positives,
# true_negatives, false_positives, false_negatives. The fields indicate the counts. They can all
# be numpy arrays of the same length. The metrics are expected to return NaN where undefined.
METRIC_DICT = {
    'selection_rate': (
        lambda x: x.predicted_positives / x.n),
    'demographic_parity': (
        lambda x: x.predicted_positives / x.n),
    'false_positive_rate': (
        lambda x: x.false_positives / x.negatives),
    'false_negative_rate': (
        lambda x: x.false_negatives / x.positives),
    'true_positive_rate': (
        lambda x: x.true_positives / x.positives),
    'true_negative_rate': (
        lambda x: x.true_negatives / x.negatives),
    'accuracy_score': ( 
        lambda x: (x.true_positives + x.true_negatives) / x.n),
    'balanced_accuracy_score': (
        lambda x: x.true_positives / x.positives + x.true_negatives / x.negatives),
}

_MATPLOTLIB_IMPORT_ERROR_MESSAGE = "Please make sure to install fairlearn[customplots] to use " \
                                   "the postprocessing plots."
