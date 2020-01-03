# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

_START_TIME = "_start_time"
_EXECUTION_TIME = "_execution_time"


class TimedExecution:
    """TimedExecution generates the python script to measure execution times.

    Specifically, it measures the time that passes from the start to the end of the
    with-statement that TimedExecution is used for.

    TimedExecution writes the code under the assumption that an Azure Machine Learning
    `azureml-core:azureml.core.Run` object called `run` exists and logs the metric
    under that run.

    script_lines = []
    script_lines.append('from azureml.core.run import Run')
    script_lines.append()"run = Run.get_context()")
    with TimedExecution("special_period", script_lines):
    #     do something

    The execution time within the with-statement is automatically tracked and saved to the
    run as a metric.
    """

    def __init__(self, procedure_name, script_lines):
        # The procedure name is used in variable names to make them unique,
        # so it cannot contain whitespace.
        assert " " not in procedure_name

        self.procedure_name = procedure_name
        self.script_lines = script_lines
        self.script_lines.append('print("Starting {}")'.format(self.procedure_name))

    def __enter__(self):
        self.script_lines.append("{} = time()".format(self.procedure_name + _START_TIME))

    def __exit__(self, type, value, traceback):  # noqa: A002
        execution_time_variable_name = self.procedure_name + _EXECUTION_TIME
        start_time_variable_name = self.procedure_name + _START_TIME
        self.script_lines.append("{} = time() - {}"
                                 .format(execution_time_variable_name, start_time_variable_name))
        self.script_lines.append("run.log('{0}', {0})"
                                 .format(execution_time_variable_name))
        self.script_lines.append('print("{} time taken: {{}}s".format({}))'
                                 .format(self.procedure_name, execution_time_variable_name))
        self.script_lines.append('print("Finished {}")'.format(self.procedure_name))
