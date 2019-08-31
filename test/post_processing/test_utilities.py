# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict, namedtuple
from itertools import permutations
import numpy as np
import pandas as pd
from fairlearn.post_processing.threshold_operation import ThresholdOperation
from fairlearn.post_processing._constants import SCORE_KEY, LABEL_KEY, ATTRIBUTE_KEY


example_attribute_names1 = ["A", "B", "C"]
example_attributes1 = [x for x in 'AAAAAAA' 'BBBBBBB' 'CCCCCC']
example_attributes2 = [x for x in 'xxxYYYY' 'xYYYYYx' 'YYYYYY']
example_labels = [int(x) for x in '0110100' '0010111' '000111']
example_scores = [int(x) for x in '0011233' '0001111' '011112']

LabelAndPrediction = namedtuple('LabelAndPrediction', 'label prediction')


def _get_grouped_data_and_base_points(attribute_value):
    data = pd.DataFrame({
        ATTRIBUTE_KEY: example_attributes1,
        SCORE_KEY: example_scores,
        LABEL_KEY: example_labels})
    grouped_data = data.groupby(ATTRIBUTE_KEY).get_group(attribute_value) \
        .sort_values(by=SCORE_KEY, ascending=False)
    x_grid = np.linspace(0, 1, 100)

    if attribute_value == "A":
        expected_roc_points = pd.DataFrame({
            "x": [0, 0.25, 0.5, 0.5, 1],
            "y": [0, 1/3,  2/3, 1,   1],
            "operation": [ThresholdOperation('>', np.inf),
                          ThresholdOperation('<', 0.5),
                          ThresholdOperation('<', 1.5),
                          ThresholdOperation('<', 2.5),
                          ThresholdOperation('>', -np.inf)]
        })
        ignore_for_base_points = [1, 2]

    if attribute_value == "B":
        expected_roc_points = pd.DataFrame({
            "x": [0, 1/3, 1],
            "y": [0, 3/4, 1],
            "operation": [ThresholdOperation('>', np.inf),
                          ThresholdOperation('<', 0.5),
                          ThresholdOperation('>', -np.inf)]
        })
        ignore_for_base_points = []

    if attribute_value == "C":
        expected_roc_points = pd.DataFrame({
            "x": [0, 0,   2/3, 1],
            "y": [0, 1/3,  1,  1],
            "operation": [ThresholdOperation('>', np.inf),
                          ThresholdOperation('<', 0.5),
                          ThresholdOperation('<', 1.5),
                          ThresholdOperation('>', -np.inf)]
        })
        ignore_for_base_points = [0]

    return grouped_data, expected_roc_points, ignore_for_base_points, x_grid


def _generate_list_reduction_permutations():
    list_reduction_permutations = []
    for permutation in permutations([0, 0, 1]):
        list_reduction_permutations.append(permutation)
    for permutation in permutations([0, 1, 1]):
        list_reduction_permutations.append(permutation)

    return list_reduction_permutations


def _generate_empty_list_permutations():
    empty_list_permutations = []

    n = len(example_attributes1)
    for permutation in permutations([0, 0, n]):
        empty_list_permutations.append(permutation)
    for permutation in permutations([0, n, n]):
        empty_list_permutations.append(permutation)

    return empty_list_permutations


def _get_discretized_predictions(adjusted_model):
    labels_and_predictions = defaultdict(list)
    for i in range(len(example_attributes1)):
        labels_and_predictions[example_attributes1[i]].append(
            LabelAndPrediction(example_labels[i],
                               adjusted_model(example_attributes1[i], example_scores[i])))

    return {
        attribute_value: [
            LabelAndPrediction(lp.label, int(lp.prediction >= 0.5))
            for lp in labels_and_predictions[attribute_value]
        ] for attribute_value in labels_and_predictions
    }
