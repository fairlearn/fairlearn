# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import EqualizedOdds, DemographicParity, GroupLossMoment

from tempeh.execution.azureml.workspace import get_workspace


class PerfTestConfiguration:
    def __init__(self, dataset, predictor, mitigator, disparity_metric):
        self.dataset = dataset
        self.predictor = predictor
        self.mitigator = mitigator
        self.disparity_metric = disparity_metric

    def __repr__(self):
        return "[dataset: {}, predictor: {}, mitigator: {}, disparity_metric: {}]" \
               .format(self.dataset, self.predictor, self.mitigator, self.disparity_metric)


def pytest_addoption(parser):
    parser.addoption("--dataset", action="store")
    parser.addoption("--predictor", action="store")
    parser.addoption("--mitigator", action="store")
    

def pytest_generate_tests(metafunc):
    if metafunc.config.option.mitigator == ExponentiatedGradient.__name__:
        disparity_metrics = [EqualizedOdds, DemographicParity]
    elif metafunc.config.option.mitigator == GridSearch.__name__:
        disparity_metrics = [EqualizedOdds, DemographicParity]
    elif metafunc.config.option.mitigator == ThresholdOptimizer.__name__:
        disparity_metrics = ["equalized_odds", "demographic_parity"]

    configurations = [PerfTestConfiguration(
        metafunc.config.option.dataset,
        metafunc.config.option.predictor,
        metafunc.config.option.mitigator,
        disparity_metric) for disparity_metric in disparity_metrics]

    metafunc.parametrize("perf_test_configuration", configurations)


@pytest.fixture(scope="session")
def workspace():
    return get_workspace()
