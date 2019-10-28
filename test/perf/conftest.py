# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch


THRESHOLD_OPTIMIZER = ThresholdOptimizer.__name__
EXPONENTIATED_GRADIENT = ExponentiatedGradient.__name__
GRID_SEARCH = GridSearch.__name__

MEMORY = "memory"
TIME = "time"

# nested by dataset - predictor - mitigation technique
# TODO try all disparity metrics!
COMBINATIONS = {
    "adult_uci": {
        "rbm_svm": {
            THRESHOLD_OPTIMIZER: {
                TIME: 13,
                MEMORY: 100000
            },
            EXPONENTIATED_GRADIENT: {
                TIME: 2600,
                MEMORY: 100000
            },
            GRID_SEARCH: {
                TIME: 1500,
                MEMORY: 100000
            }
        },
        "decision_tree_classifier": {
            THRESHOLD_OPTIMIZER: {
                TIME: 2,
                MEMORY: 100000
            },
            EXPONENTIATED_GRADIENT: {
                TIME: 50,
                MEMORY: 100000
            },
            GRID_SEARCH: {
                TIME: 30,
                MEMORY: 100000
            }
        }
    },
    'compas': {
        "rbm_svm": {
            THRESHOLD_OPTIMIZER: {
                TIME: 3,
                MEMORY: 100000
            },
            EXPONENTIATED_GRADIENT: {
                TIME: 150,
                MEMORY: 100000
            },
            # no grid search since there are more than two sensitive feature values
        },
        "decision_tree_classifier": {
            THRESHOLD_OPTIMIZER: {
                TIME: 2.5,
                MEMORY: 100000
            },
            EXPONENTIATED_GRADIENT: {
                TIME: 12,
                MEMORY: 100000
            },
            # no grid search since there are more than two sensitive feature values
        }
    }
}


class PerfTestConfiguration:
    def __init__(self, dataset, predictor, mitigator, max_time_consumption,
                 max_memory_consumption):
        self.dataset = dataset
        self.predictor = predictor
        self.mitigator = mitigator
        self.max_time_consumption = max_time_consumption
        self.max_memory_consumption = max_memory_consumption

    def __repr__(self):
        return "[dataset: {}, predictor: {}, mitigator: {}, max_time_consumption: {}, " \
               "max_memory_consumption: {}]".format(self.dataset, self.predictor,
                                                    self.mitigator, self.max_time_consumption,
                                                    self.max_memory_consumption)


def get_all_perf_test_configurations():
    perf_test_configurations = []
    for dataset in COMBINATIONS:
        for predictor in COMBINATIONS[dataset]:
            for mitigator in COMBINATIONS[dataset][predictor]:
                max_time_consumption = COMBINATIONS[dataset][predictor][mitigator][TIME]
                max_memory_consumption = COMBINATIONS[dataset][predictor][mitigator][MEMORY]
                perf_test_configurations.append(
                    PerfTestConfiguration(dataset, predictor, mitigator, max_time_consumption,
                                          max_memory_consumption))

    return perf_test_configurations
