# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch

from timed_execution import TimedExecution, _EXECUTION_TIME


_MITIGATION = "mitigation"
_ESTIMATOR_FIT = 'estimator_fit'


def generate_script(request, perf_test_configuration, script_name, script_directory):
    if not os.path.exists(script_directory):
        os.makedirs(script_directory)

    script_lines = []
    add_imports(script_lines)
    script_lines.append("")
    script_lines.append("run = Run.get_context()")
    add_dataset_setup(script_lines, perf_test_configuration)
    add_unconstrained_estimator_fitting(script_lines, perf_test_configuration)
    add_mitigation(script_lines, perf_test_configuration)
    add_additional_metric_calculation(script_lines)
    script_lines.append("")

    print("\n\n{}\n\n".format("="*100))

    with open(os.path.join(script_directory, script_name), 'w') as script_file:  # noqa: E501
        script_file.write("\n".join(script_lines))


def add_imports(script_lines):
    script_lines.append('from time import time')
    script_lines.append('from tempeh.configurations import models, datasets')
    script_lines.append('from fairlearn.postprocessing import ThresholdOptimizer')
    script_lines.append('from fairlearn.reductions import ExponentiatedGradient, GridSearch')
    script_lines.append('from fairlearn.reductions import DemographicParity, EqualizedOdds')
    script_lines.append('from azureml.core.run import Run')


def add_dataset_setup(script_lines, perf_test_configuration):
    script_lines.append('print("Downloading dataset")')
    script_lines.append('dataset = datasets["{}"]()'.format(perf_test_configuration.dataset))
    script_lines.append('X_train, X_test = dataset.get_X()')
    script_lines.append('y_train, y_test = dataset.get_y()')
    script_lines.append('print("Done downloading dataset")')

    if perf_test_configuration.dataset == "adult_uci":
        # sensitive feature is 8th column (sex)
        script_lines.append('sensitive_features_train = X_train[:, 7]')
        script_lines.append('sensitive_features_test = X_test[:, 7]')
    elif perf_test_configuration.dataset == "diabetes_sklearn":
        # sensitive feature is 2nd column (sex)
        # features have been scaled, but sensitive feature needs to be str or int
        script_lines.append('sensitive_features_train = X_train[:, 1].astype(str)')
        script_lines.append('sensitive_features_test = X_test[:, 1].astype(str)')
        # labels can't be floats as of now
        script_lines.append('y_train = y_train.astype(int)')
        script_lines.append('y_test = y_test.astype(int)')
    elif perf_test_configuration.dataset == "compas":
        # sensitive feature is either race or sex
        # TODO add another case where we use sex as well, or both (?)
        script_lines.append('sensitive_features_train, sensitive_features_test = dataset.get_sensitive_features("race")')
        script_lines.append('y_train = y_train.astype(int)')
        script_lines.append('y_test = y_test.astype(int)')
    else:
        raise ValueError("Sensitive features unknown for dataset {}"
                         .format(perf_test_configuration.dataset))


def add_unconstrained_estimator_fitting(script_lines, perf_test_configuration):
    with TimedExecution(_ESTIMATOR_FIT, script_lines):
        script_lines.append('estimator = models["{}"]()'.format(perf_test_configuration.predictor))
        script_lines.append('unconstrained_predictor = models["{}"]()'.format(perf_test_configuration.predictor))
        script_lines.append('unconstrained_predictor.fit(X_train, y_train)')


def add_mitigation(script_lines, perf_test_configuration):
    with TimedExecution(_MITIGATION, script_lines):
        if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
            script_lines.append('mitigator = ThresholdOptimizer('
                                'unconstrained_predictor=unconstrained_predictor, '
                                'constraints="{}")'.format(perf_test_configuration.disparity_metric))
        elif perf_test_configuration.mitigator == ExponentiatedGradient.__name__:
            script_lines.append('mitigator = ExponentiatedGradient('
                                'estimator=estimator, '
                                'constraints={}())'.format(perf_test_configuration.disparity_metric))
        elif perf_test_configuration.mitigator == GridSearch.__name__:
            script_lines.append('mitigator = GridSearch(estimator=estimator, '
                                'constraints={}())'.format(perf_test_configuration.disparity_metric))
        else:
            raise Exception("Unknown mitigation technique.")

        script_lines.append('mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)')

        if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
            script_lines.append('mitigator.predict('
                                'X_test, '
                                'sensitive_features=sensitive_features_test, '
                                'random_state=1)')
        else:
            script_lines.append('predictions = mitigator.predict(X_test)')


def add_additional_metric_calculation(script_lines, perf_test_configuration):
    # We need to know how much overhead fairlearn adds on top of the basic estimator fit.
    estimator_fit_time_variable_name = _ESTIMATOR_FIT + _EXECUTION_TIME
    mitigation_time_variable_name = _MITIGATION + _EXECUTION_TIME

    additional_metrics = {
        'mitigation_time_overhead_absolute': '-',
        'mitigation_time_overhead_relative': '/'
    }
    for metric_name, operator in additional_metrics.items():
        script_lines.append("{} = {} {} {}"
                            .format(metric_name, mitigation_time_variable_name, operator,
                                    estimator_fit_time_variable_name))
        script_lines.append("run.log('{0}', {0})".format(metric_name))

    # ExponentiatedGradient tells us how many oracle calls were made.
    if perf_test_configuration.mitigatior == ExponentiatedGradient.__name__:
        script_lines.append("run.log('n_oracle_calls', mitigator._expgrad_result.n_oracle_calls)")
