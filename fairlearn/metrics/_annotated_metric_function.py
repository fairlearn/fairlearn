# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import Callable, Dict, List

import pandas as pd


class AnnotatedMetricFunction:
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
