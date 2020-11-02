# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import numpy as np
from typing import Callable, List, Union

from ._metric_frame import MetricFrame

transform_options = [
    'difference',
    'group_min',
    'group_max',
    'ratio',
]


class _DerivedMetric:
    def __init__(self,
                 *,
                 metric: Callable[..., Union[float, int]],
                 transform: str,
                 sample_param_names: List[str]):
        assert transform in transform_options
        self._transform = transform

        assert callable(metric)
        self._metric_fn = metric

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
                        sample_param_names: List[str]) -> Callable[..., Union[float, int]]:
    """Create a scalar returning metric function based on aggregation of a disaggregated metric.

    Many higher order machine learning operations (such as hyperparameter tuning)
    make use of functions which return scalar metrics. We can create such a function
    for our disaggregated metrics with this function.
    """
    dm = _DerivedMetric(metric=metric,
                        transform=transform,
                        sample_param_names=sample_param_names)
    return dm
