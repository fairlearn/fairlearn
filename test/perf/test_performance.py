# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import pytest
from time import time
from tempeh.configurations import models, datasets

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.postprocessing._threshold_optimizer import DEMOGRAPHIC_PARITY
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import DemographicParity

from conftest import get_all_perf_test_configurations

all_perf_test_configurations = get_all_perf_test_configurations()
all_perf_test_configurations_descriptions = \
    [config.__repr__().replace(' ', '') for config in all_perf_test_configurations]


logging.basicConfig(level=logging.DEBUG)


@pytest.mark.parametrize("perf_test_configuration", all_perf_test_configurations,
                         ids=all_perf_test_configurations_descriptions)
def test_perf(perf_test_configuration, request):
    print("Starting with test case {}".format(request.node.name))
    print("Downloading dataset")
    dataset = datasets[perf_test_configuration.dataset]()
    print("Done downloading dataset")

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
        dataset.y_train = dataset.y_train.astype(int)
        dataset.y_test = dataset.y_test.astype(int)
    else:
        raise ValueError("Sensitive features unknown for dataset {}"
                         .format(perf_test_configuration.dataset))

    print("Fitting estimator")
    estimator = models[perf_test_configuration.predictor]()
    unconstrained_predictor = models[perf_test_configuration.predictor]()
    unconstrained_predictor.fit(dataset.X_train, dataset.y_train)
    print("Done fitting estimator")

    start_time = time()
    if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
        mitigator = ThresholdOptimizer(unconstrained_predictor=unconstrained_predictor,
                                       constraints=DEMOGRAPHIC_PARITY)
    elif perf_test_configuration.mitigator == ExponentiatedGradient.__name__:
        mitigator = ExponentiatedGradient(estimator=estimator,
                                          constraints=DemographicParity())
    elif perf_test_configuration.mitigator == GridSearch.__name__:
        mitigator = GridSearch(estimator=estimator,
                               constraints=DemographicParity())
    else:
        raise Exception("Unknown mitigation technique.")

    print("Fitting mitigator")

    mitigator.fit(dataset.X_train, dataset.y_train, sensitive_features=sensitive_features_train)

    if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
        mitigator.predict(dataset.X_test, sensitive_features=sensitive_features_test,
                          random_state=1)
    else:
        mitigator.predict(dataset.X_test)

    # TODO evaluate accuracy/fairness tradeoff

    total_time = time() - start_time
    print("Total time taken: {}s".format(total_time))
    print("Maximum allowed time: {}s".format(perf_test_configuration.max_time_consumption))
    assert total_time <= perf_test_configuration.max_time_consumption
    print("\n\n===============================================================\n\n")
