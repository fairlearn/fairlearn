# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import Dict, List, Callable

import pandas as pd


class MetricFunctionRequest:
    """Wrapper for functions, to give them args and kwargs."""

    def __init__(self,
                 *,
                 func: Callable,
                 args: List[str] = None,
                 kwargs: Dict[str, str] = None):
        self._func = func
        self._args = ['y_true', 'y_pred']
        if args is not None:
            self._args = args
        self._kwargs = dict()
        if kwargs is not None:
            self._kwargs = kwargs

    @property
    def func(self) -> Callable:
        """Return the wrapped function."""
        return self._func

    @property
    def args(self) -> List[str]:
        """Return the list of positional arguments."""
        return self._args

    @property
    def kwargs(self) -> Dict[str, str]:
        """Return the mapping from column names to kwarg names."""
        return self._kwargs

    def invoke(self, df: pd.DataFrame):
        """Invoke the wrapped function on the supplied DataFrame."""
        args = [df[arg_name] for arg_name in self.args]

        kwargs = dict()
        for func_arg_name, data_arg_name in self.kwargs.items():
            kwargs[func_arg_name] = df[data_arg_name]

        result = self.func(*args, **kwargs)

        return result


def apply_to_dataframe(
        data: pd.DataFrame,
        metric_functions: Dict[str, MetricFunctionRequest]) -> pd.Series:
    """Apply metric functions to a DataFrame."""
    values = dict()
    for name, mf in metric_functions.items():
        values[name] = mf.invoke(data)
    result = pd.Series(data=values.values(), index=values.keys())
    return result


class MFv2:
    """An alternative for MetricFrame."""

    def __init__(self,
                 *,
                 metric_functions: Dict[str, MetricFunctionRequest],
                 data: pd.DataFrame,
                 sensitive_features: List[str],
                 control_features: List[str] = None
                 ):
        if control_features is None:
            control_features = []

        if len(control_features) == 0:
            self._overall = apply_to_dataframe(data, metric_functions=metric_functions)
        else:
            self._overall = data.groupby(by=control_features).apply(
                apply_to_dataframe, metric_functions=metric_functions
            )

        self._by_group = data.groupby(control_features+sensitive_features).apply(
            apply_to_dataframe, metric_functions=metric_functions)

    @property
    def overall(self):
        """Return the overall value."""
        return self._overall

    @property
    def by_group(self):
        """Return the value for each subgroup."""
        return self._by_group
