# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from typing import Callable, Dict, List, Optional

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_NAME = "metric"

_METRIC_FUNCTION_NONE = "Found 'None' instead of metric function"
_METRIC_FUNCTION_NOT_CALLABLE = "Object passed as metric function not callable"


class AnnotatedMetricFunction:
    """Wraps functions to make them callable with a DataFrame argument.
    
    The :class:`MetricFrame` makes extensive use of `pandas` DataFrames
    internally. In particular, combinations of sensitive (and control)
    features are selected using `DataFrame.groupby()` and acted on via
    `DataFrame.apply()`.
    """

    def __init__(
        self,
        *,
        func: Callable,
        name: Optional[str],
        postional_argument_names: List[str] = None,
        kw_argument_mapping: Dict[str, str] = None
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
                logger.warning("Supplied 'func' had no __name__ attribute")
                self.name = _DEFAULT_NAME
        else:
            self.name = name

        self.func = func
        self.postional_argument_names = ["y_true", "y_pred"]
        if postional_argument_names is not None:
            self.postional_argument_names = postional_argument_names
        self.kw_argument_mapping = dict()
        if kw_argument_mapping is not None:
            self.kw_argument_mapping = kw_argument_mapping

    def invoke(self, df: pd.DataFrame):
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
        These are subject to some extra list-washing; columns requiring this
        treatment are recorded in `multi_d_columns`.
        """
        args = []
        for arg_name in self.postional_argument_names:
            # Need to convert to list first in case we have 2D arrays
            args.append(np.asarray(list(df[arg_name])))

        kwargs = dict()
        for func_arg_name, data_arg_name in self.kw_argument_mapping.items():
            # Need to convert to list first in case we have 2D arrays
            kwargs[func_arg_name] = np.asarray(list(df[data_arg_name]))

        result = self.func(*args, **kwargs)

        return result
