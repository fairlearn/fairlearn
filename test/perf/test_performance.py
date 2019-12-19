# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import os
import pytest
import time

from azureml.core import Experiment, RunConfiguration, ScriptRunConfig

from tempeh.execution.azureml.environment_setup import configure_environment

from conftest import get_all_perf_test_configurations
from environment_setup import build_package
from script_generation import generate_script

all_perf_test_configurations = get_all_perf_test_configurations()
all_perf_test_configurations_descriptions = \
    [config.__repr__().replace(' ', '') for config in all_perf_test_configurations]

SCRIPT_DIRECTORY = os.path.join('test', 'perf', 'scripts')
EXPERIMENT_NAME = 'perftest'

logging.basicConfig(level=logging.DEBUG)


# ensure the tests are run from the fairlearn repository base directory
if not os.getcwd().endswith("fairlearn"):
    if not os.path.exists("test") or not os.path.exists("fairlearn"):
        raise Exception("Please run perf tests from the fairlearn repository base directory. "
                        "Current working directory: {}".format(os.getcwd()))


@pytest.mark.parametrize("perf_test_configuration", all_perf_test_configurations,
                         ids=all_perf_test_configurations_descriptions)
def test_perf(perf_test_configuration, workspace, request):
    print("Starting with test case {}".format(request.node.name))

    script_name = determine_script_name(request.node.name)
    generate_script(request, perf_test_configuration, script_name, SCRIPT_DIRECTORY)
    wheel_file = build_package()

    experiment = Experiment(workspace=workspace, name=EXPERIMENT_NAME)
    compute_target = workspace.get_default_compute_target(type='cpu')
    run_config = RunConfiguration()
    run_config.target = compute_target

    environment = configure_environment(workspace, wheel_file=wheel_file)
    run_config.environment = environment
    environment.register(workspace=workspace)
    script_run_config = ScriptRunConfig(source_directory=SCRIPT_DIRECTORY,
                                        script=script_name,
                                        run_config=run_config)
    print("submitting run")
    run = experiment.submit(config=script_run_config)
    print("submitted run")
    run.wait_for_completion()
    print("run completed")
    while run.get_status() == "Finalizing":
        time.sleep(1)
    assert run.get_status() == "Completed"


def determine_script_name(test_case_name):
    hashed_test_case_name = hash(test_case_name)
    hashed_test_case_name = hashed_test_case_name if hashed_test_case_name >= 0 \
        else -1 * hashed_test_case_name
    return "{}.py".format(str(hashed_test_case_name))
