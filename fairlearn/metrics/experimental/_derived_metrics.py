# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

from ._grouped_metric import GroupedMetric

aggregate_options = [
    'difference',
    'difference_to_overall'
]


class _DerivedMetric:
    def __init__(self, aggregate, metric_fn, sample_param_names):
        assert aggregate in aggregate_options
        self._aggregate = aggregate

        assert callable(metric_fn)
        self._metric_fn = metric_fn

        self._sample_param_names = sample_param_names

    def __call__(self, y_true, y_pred, *, sensitive_features, **all_params):
        params = dict()
        sample_params = dict()
        for k, v in all_params.items():
            if k in self._sample_param_names:
                sample_params[k] = v
            else:
                params[k] = v

        all_metrics = GroupedMetric(self._metric_fn,
                                    y_true, y_pred,
                                    sensitive_features=sensitive_features,
                                    sample_params=sample_params,
                                    params=params)

        result = np.nan
        if self._aggregate == 'difference':
            result = all_metrics.difference().iloc[0, 0]
        elif self._aggregate == 'difference_to_overall':
            result = all_metrics.difference(method='to_overall').iloc[0, 0]
        else:
            raise ValueError("Cannot get here")

        return result


def make_derived_metric(aggregate, metric_fn, sample_param_names=None):
    """Read a placeholder comment."""
    return _DerivedMetric(aggregate, metric_fn, sample_param_names)
