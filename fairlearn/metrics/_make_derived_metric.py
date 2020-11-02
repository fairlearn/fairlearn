# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import inspect
import numpy as np
from typing import Callable, List, Union

from ._metric_frame import MetricFrame

transform_options = [
    'difference',
    'group_min',
    'group_max',
    'ratio',
]

_METRIC_CALLABLE_ERROR = "Supplied metric object must be callable"
_METHOD_ARG_ERROR = "Callables which accept a 'method' argument " \
    "may not be passed to make_derived_metric(). Please use functools.partial()"
_INVALID_TRANSFORM = "Transform must be one of {0}".format(transform_options)


class _DerivedMetric:
    def __init__(self,
                 *,
                 metric: Callable[..., Union[float, int]],
                 transform: str,
                 sample_param_names: List[str]):

        if not callable(metric):
            raise ValueError(_METRIC_CALLABLE_ERROR)
        sig = inspect.signature(metric)
        if 'method' in sig.parameters:
            raise ValueError(_METHOD_ARG_ERROR)
        self._metric_fn = metric

        if transform not in transform_options:
            raise ValueError(_INVALID_TRANSFORM)
        self._transform = transform

        self._sample_param_names = []
        if sample_param_names is not None:
            self._sample_param_names = sample_param_names

    def __call__(self,
                 y_true,
                 y_pred,
                 *,
                 sensitive_features,
                 method=None,
                 **other_params) -> Union[float, int]:
        sample_params = dict()
        params = dict()
        for k, v in other_params.items():
            if k in self._sample_param_names:
                sample_params[k] = v
            else:
                params[k] = v

        dispatch_fn = functools.partial(self._metric_fn, **params)

        all_metrics = MetricFrame(dispatch_fn,
                                  y_true, y_pred,
                                  sensitive_features=sensitive_features,
                                  sample_params=sample_params)

        result = np.nan
        if self._transform == 'difference':
            result = all_metrics.difference(method=method)
        elif self._transform == 'ratio':
            result = all_metrics.ratio(method=method)
        elif self._transform == 'group_min':
            result = all_metrics.group_min()
        elif self._transform == 'group_max':
            result = all_metrics.group_max()
        else:
            raise ValueError("Cannot get here")

        return result


def make_derived_metric(*,
                        metric: Callable[..., Union[float, int]],
                        transform: str,
                        sample_param_names: List[str] = None) -> Callable[..., Union[float, int]]:
    """Create a scalar returning metric function based on aggregation of a disaggregated metric.

    Many higher order machine learning operations (such as hyperparameter tuning)
    make use of functions which return scalar metrics. We can create such a function
    for our disaggregated metrics with this function.

    This function takes a metric function, a string to specify the desired aggregation
    transform (matching the methods on :class:`.MetricFrame`), and a list of
    parameter names to treat as sample parameters.

    The result is a callable object which has the same signature as the original
    function, with two arguments added. These are :code:`sensitive_features=`, to 
    specify the subgroups in the dataset, and :code:`method=`, if required by the
    selected aggregation transform. The result of this function is identical to
    creating a :class:`.MetricFrame` object, and then calling the method specified
    by the :code:`transform=` argument (with the :code:`method=` argument, if
    required).

    Parameters
    ----------
    metric : callable
        The metric function from which the new function should be derived

    transform : str
        Selects the transformation aggregation the resultant function should use

    sample_param_names : List[str]
        A list of parameters names of the underlying :code:`metric` which should
        be treated as sample parameters (i.e. the same leading dimension as the
        :code:`y_true` and :code:`y_pred` parameters)

    Returns
    -------
    callable
        Function with the same signature as the :code:`metric` but with additional
        :code:`sensitive_feature=` and :code:`method=` arguments, to enable the
        required computation
    """
    dm = _DerivedMetric(metric=metric,
                        transform=transform,
                        sample_param_names=sample_param_names)
    return dm
