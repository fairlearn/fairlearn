# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np

from ._grouped_metric import GroupedMetric

aggregate_options = [
    'difference',
    'difference_to_overall',
    'ratio',
    'ratio_to_overall',
]


class _DerivedMetric:
    def __init__(self, aggregate, metric_fn):
        assert aggregate in aggregate_options
        self._aggregate = aggregate

        assert callable(metric_fn)
        self._metric_fn = metric_fn

    def __call__(self, y_true, y_pred, *, sensitive_features, **sample_params):

        all_metrics = GroupedMetric(self._metric_fn,
                                    y_true, y_pred,
                                    sensitive_features=sensitive_features,
                                    sample_params=sample_params)

        result = np.nan
        if self._aggregate == 'difference':
            result = all_metrics.difference(method='minmax')[self._metric_fn.__name__]
        elif self._aggregate == 'difference_to_overall':
            result = all_metrics.difference(method='to_overall')[self._metric_fn.__name__]
        elif self._aggregate == 'ratio':
            result = all_metrics.ratio(method='minmax')[self._metric_fn.__name__]
        elif self._aggregate == 'ratio_to_overall':
            result = all_metrics.ratio(method='to_overall')[self._metric_fn.__name__]
        else:
            raise ValueError("Cannot get here")

        return result


def make_derived_metric(aggregate, metric_fn):
    """Read a placeholder comment."""
    return _DerivedMetric(aggregate, metric_fn)
