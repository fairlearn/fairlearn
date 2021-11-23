# Copyright (c) Fairlearn contributors.
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

    def invoke(self, df: pd.DataFrame, split_columns: Dict[str, List[str]]):
        """Invoke the wrapped function on the supplied DataFrame."""
        args = []
        for arg_name in self.args:
            if arg_name in split_columns:
                sub_frame = df[split_columns[arg_name]]
                args.append(sub_frame.to_numpy())
            else:
                args.append(df[arg_name])

        kwargs = dict()
        for func_arg_name, data_arg_name in self.kwargs.items():
            if data_arg_name in split_columns:
                sub_frame = df[split_columns[data_arg_name]]
                kwargs[func_arg_name] = sub_frame.to_numpy()
            else:
                kwargs[func_arg_name] = df[data_arg_name]

        result = self.func(*args, **kwargs)

        return result
