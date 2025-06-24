# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import logging
import warnings
from typing import Callable, Generic, Tuple, TypeVar

import narwhals.stable.v1 as nw
import numpy as np
from narwhals.typing import IntoDataFrame

R = TypeVar("R")

logger = logging.getLogger(__name__)

_DEFAULT_NAME = "metric"

_METRIC_FUNCTION_NONE = "Found 'None' instead of metric function"
_METRIC_FUNCTION_NOT_CALLABLE = "Object passed as metric function not callable"


class AnnotatedMetricFunction(Generic[R]):
    """Wraps functions to make them callable with a DataFrame argument.

    The :class:`MetricFrame` makes extensive use of `pandas` DataFrames
    internally. In particular, combinations of sensitive (and control)
    features are selected using `DataFrame.groupby()` and acted on via
    `DataFrame.apply()`. The net result of this is that it is useful
    to have a function wrapper which can be invoked with a DataFrame
    argument, and extract its arguments from that DataFrame.

    Parameters
    ----------
    func : callable
        The metric function we wish to invoke
    name: str | None
        An optional string defining the name of the function
    positional_argument_names: list[str] | None
        The column names to be extracted and passed as positional arguments
        when invoking the function
    kw_argument_mapping: dict[str, str] | None
        The column names which are to be passed as keyword arguments
        when invoking the function. Since the DataFrame column names may
        not match the function's argument names, this is a dictionary
        where the keys are the function argument names, and the values
        are the column names
    """

    def __init__(
        self,
        *,
        func: Callable[[*Tuple[np.ndarray, ...]], R],
        name: str | None = None,
        positional_argument_names: list[str] | None = None,
        kw_argument_mapping: dict[str, str] | None = None,
    ):
        if func is None:
            raise ValueError(_METRIC_FUNCTION_NONE)
        if not callable(func):
            raise ValueError(_METRIC_FUNCTION_NOT_CALLABLE)
        self.func = func

        if name is None:
            if hasattr(func, "__name__"):
                self.name = func.__name__
            else:
                warnings.warn("Supplied 'func' had no __name__ attribute")
                self.name = _DEFAULT_NAME
        else:
            self.name = name

        self.postional_argument_names = ["y_true", "y_pred"]
        if positional_argument_names is not None:
            self.postional_argument_names = positional_argument_names
        self.kw_argument_mapping = dict()
        if kw_argument_mapping is not None:
            self.kw_argument_mapping = kw_argument_mapping

    def __call__(self, df: IntoDataFrame) -> R:
        """Invoke the wrapped function on the supplied DataFrame.

        The function extracts its arguments from the supplied DataFrame :code:`df`.
        Columns listed in :code:`self.postional_argument_names` are supplied
        positionally, while those
        in :code:`self.kw_argument_mapping` are supplied as keyword arguments.

        There are two subtleties.
        Firstly, the keyword arguments expected by the function might not
        have the same names as the columns in the supplied DataFrame.
        This is the reason :code:`self.kw_argument_mapping` is a dictionary, rather than
        just a list of column names.

        The second issue is coping with when users have passed in a 2D array as
        a named argument (especially, `y_true` or `y_pred`).
        For this reason, we perform an extra `np.stack` operation, to make sure the
        expected types are passed to the underlying metric function.
        """
        df_nw = nw.from_native(df, eager_only=True, pass_through=False)
        args = [
            np.stack(df_nw.get_column(arg_name).to_numpy())
            for arg_name in self.postional_argument_names
        ]
        kwargs = {
            func_arg_name: np.stack(df_nw.get_column(data_arg_name).to_numpy())
            for func_arg_name, data_arg_name in self.kw_argument_mapping.items()
        }
        return self.func(*args, **kwargs)
