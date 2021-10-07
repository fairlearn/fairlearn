# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import Dict, List, Callable

import pandas as pd


class MetricFunctionRequest:
    def __init__(self, func: Callable, arguments: List[str] = ['y_true', 'y_pred']):
        self._func = func
        self._args = arguments

    @property
    def func(self) -> Callable:
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


def apply_to_dataframe(data: pd.DataFrame, metric_functions: Dict[str, MetricFunctionRequest]) -> pd.Series:
    values = dict()
    for name, mf in metric_functions.items():
        values[name] = mf.invoke(data)
    result = pd.Series(data=values.values(), index=values.keys())
    return result


class MFv2:
    def __init__(self,
                 metric_functions: Dict[str, MetricFunctionRequest],
                 data: pd.DataFrame,
                 sensitive_features: List[str]
                 ):
        self._overall = apply_to_dataframe(data, metric_functions=metric_functions)

        self._by_group = data.groupby(sensitive_features).apply(
            apply_to_dataframe, metric_functions=metric_functions)

    @property
    def overall(self):
        return self._overall

    @property
    def by_group(self):
        return self._by_group
