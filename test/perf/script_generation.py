# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, GridSearch


def generate_script(request, perf_test_configuration, script_name, script_directory):
    if not os.path.exists(script_directory):
        os.makedirs(script_directory)

    script_lines = []
    # imports
    script_lines.append('from time import time')
    script_lines.append('from tempeh.configurations import models, datasets')
    script_lines.append('from fairlearn.postprocessing._threshold_optimizer import DEMOGRAPHIC_PARITY')
    script_lines.append('from fairlearn.postprocessing import ThresholdOptimizer')
    script_lines.append('from fairlearn.reductions import ExponentiatedGradient, GridSearch')
    script_lines.append('from fairlearn.reductions import DemographicParity')
    script_lines.append('from azureml.core.run import Run')

    script_lines.append("")

    # logic
    script_lines.append("run = Run.get_context()")
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

    script_lines.append('print("Fitting estimator")')
    script_lines.append('estimator = models["{}"]()'.format(perf_test_configuration.predictor))
    script_lines.append('unconstrained_predictor = models["{}"]()'.format(perf_test_configuration.predictor))
    script_lines.append('unconstrained_predictor.fit(X_train, y_train)')
    script_lines.append('print("Done fitting estimator")')

    script_lines.append('start_time = time()')
    if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
        script_lines.append('mitigator = ThresholdOptimizer('
                            'unconstrained_predictor=unconstrained_predictor,'
                            'constraints=DEMOGRAPHIC_PARITY)')
    elif perf_test_configuration.mitigator == ExponentiatedGradient.__name__:
        script_lines.append('mitigator = ExponentiatedGradient('
                            'estimator=estimator,'
                            'constraints=DemographicParity())')
    elif perf_test_configuration.mitigator == GridSearch.__name__:
        script_lines.append('mitigator = GridSearch(estimator=estimator,'
                            'constraints=DemographicParity())')
    else:
        raise Exception("Unknown mitigation technique.")

    script_lines.append('print("Fitting mitigator")')
    script_lines.append('mitigator.fit(X_train, y_train, sensitive_features=sensitive_features_train)')

    if perf_test_configuration.mitigator == ThresholdOptimizer.__name__:
        script_lines.append('mitigator.predict('
                            'X_test, '
                            'sensitive_features=sensitive_features_test,'
                            'random_state=1)')
    else:
        script_lines.append('mitigator.predict(X_test)')

    # TODO evaluate accuracy/fairness tradeoff

    script_lines.append('total_time = time() - start_time')
    script_lines.append('run.log("total_time", total_time)')
    script_lines.append('print("Total time taken: {}s".format(total_time))')
    print("\n\n===============================================================\n\n")

    with open(os.path.join(script_directory, script_name), 'w') as script_file:  # noqa: E501
        script_file.write("\n".join(script_lines))
