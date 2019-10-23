# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
import pytest
from time import time
from tempeh.configurations import models, datasets

from fairlearn.post_processing import ThresholdOptimizer
from fairlearn.post_processing.threshold_optimizer import DEMOGRAPHIC_PARITY
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions.moments import DemographicParity
from fairlearn.reductions.grid_search.simple_quality_metrics import SimpleClassificationQualityMetric

from conftest import get_all_perf_test_configurations

all_perf_test_configurations = get_all_perf_test_configurations()
all_perf_test_configurations_descriptions = \
    [config.__repr__().replace(' ', '') for config in all_perf_test_configurations]


logging.basicConfig(level=logging.DEBUG)

@pytest.mark.parametrize("perf_test_configuration", all_perf_test_configurations,
                         ids=all_perf_test_configurations_descriptions)
def test_perf(perf_test_configuration):
    print("Downloading dataset")
    dataset = datasets[perf_test_configuration.dataset]()
    print("Downloaded dataset")

    if perf_test_configuration.dataset == "adult_uci":
        # sensitive feature is 8th column (sex)
        sensitive_features_train = dataset.X_train[:, 7]
        sensitive_features_test = dataset.X_test[:, 7]
    elif perf_test_configuration.dataset == "diabetes_sklearn":
        # sensitive feature is 2nd column (sex)
        # features have been scaled, but sensitive feature needs to be str or int
        sensitive_features_train = dataset.X_train[:, 1].astype(str)
        sensitive_features_test = dataset.X_test[:, 1].astype(str)
        # labels can't be floats as of now
        dataset.y_train = dataset.y_train.astype(int)
        dataset.y_test = dataset.y_test.astype(int)
    elif perf_test_configuration.dataset == "compas":
        # sensitive feature is either race or sex
        # TODO add another case where we use sex as well, or both (?)
        sensitive_features_train = dataset.race_train
        sensitive_features_test = dataset.race_test
    else:
        raise ValueError("Sensitive features unknown for dataset {}"
                         .format(perf_test_configuration.dataset))

    print("Fitting estimator")
    unconstrained_estimator = models[perf_test_configuration.predictor]()
    unconstrained_predictor = models[perf_test_configuration.predictor]()
    unconstrained_predictor.fit(dataset.X_train, dataset.y_train)

    start_time = time()
    if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
        mitigator = ThresholdOptimizer(unconstrained_model=unconstrained_predictor,
                                       parity_criteria=DEMOGRAPHIC_PARITY,
                                       random_state=1)
    elif perf_test_configuration.mitigator == ExponentiatedGradient.__name__:
        mitigator = ExponentiatedGradient(estimator=unconstrained_estimator,
                                          constraints=DemographicParity())
    elif perf_test_configuration.mitigator == GridSearch.__name__:
        mitigator = GridSearch(learner=unconstrained_estimator,
                               disparity_metric=DemographicParity(),
                               quality_metric=SimpleClassificationQualityMetric())
    else:
        raise Exception("Unknown mitigation technique.")

    print("Fitting mitigator")
    
    print(dataset.X_train.shape)
    print(dataset.y_train.shape)

    mitigator.fit(dataset.X_train, dataset.y_train, sensitive_features=sensitive_features_train)

    if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
        mitigator.predict(dataset.X_test, sensitive_features=sensitive_features_test)
    else:
        mitigator.predict(dataset.X_test)

    # TODO evaluate accuracy/fairness tradeoff

    total_time = time() - start_time
    assert total_time <= perf_test_configuration.max_time_consumption
    print("Total time taken: {}s".format(total_time))
    print("Maximum allowed time: {}s".format(perf_test_configuration.max_time_consumption))
