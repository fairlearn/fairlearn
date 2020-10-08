# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import numpy as np

from ._grouped_metric import GroupedMetric

aggregate_options = [
    'difference',
    'group_min',
    'group_max',
    'ratio',
]


class _DerivedMetric:
    def __init__(self, aggregate, metric_fn, sample_param_names):
        assert aggregate in aggregate_options
        self._aggregate = aggregate

        assert callable(metric_fn)
        self._metric_fn = metric_fn

        self._sample_param_names = []
        if sample_param_names is not None:
            self._sample_param_names = sample_param_names

    def __call__(self, y_true, y_pred, *, sensitive_features, method=None, **other_params):
        metric_name = 'computed_metric'
        sample_params = dict()
        params = dict()
        for k, v in other_params.items():
            if k in self._sample_param_names:
                sample_params[k] = v
            else:
                params[k] = v

        dispatch_fn = functools.partial(self._metric_fn, **params)
        dispatch_fn.__name__ = metric_name

        all_metrics = GroupedMetric(dispatch_fn,
                                    y_true, y_pred,
                                    sensitive_features=sensitive_features,
                                    sample_params=sample_params)

        result = np.nan
        if self._aggregate == 'difference':
            result = all_metrics.difference(method=method)[metric_name]
        elif self._aggregate == 'ratio':
            result = all_metrics.ratio(method=method)[metric_name]
        elif self._aggregate == 'group_min':
            result = all_metrics.group_min()[metric_name]
        elif self._aggregate == 'group_max':
            result = all_metrics.group_max()[metric_name]
        else:
            raise ValueError("Cannot get here")

        return result


def make_derived_metric(aggregate, metric_fn, sample_param_names):
    """Read a placeholder comment."""
    return _DerivedMetric(aggregate, metric_fn, sample_param_names)
