# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import List

import pandas as pd


class MetricFunctionRequest:
    def __init__(self, func, arguments: List[str] = ['y_true', 'y_pred']):
        self._func = func
        self._args = arguments

    @property
    def func(self):
        return self._func

    @property
    def arguments(self) -> List[str]:
        return self._args

    def invoke(self, df: pd.DataFrame):
        kwargs = dict()
        for arg_name in self.arguments:
            kwargs[arg_name] = df[arg_name]

        result = self.func(**kwargs)

        return result
