# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import functools
import inspect
from typing import Callable, List, Union

from ._metric_frame import MetricFrame

transform_options = [
    "difference",
    "group_min",
    "group_max",
    "ratio",
]

parameters_for_transforms = ["method"]

_METRIC_CALLABLE_ERROR = "Supplied metric object must be callable"
_METHOD_ARG_ERROR = (
    "Callables which accept a '{0}' argument "
    "may not be passed to make_derived_metric(). Please use functools.partial()"
)
_INVALID_TRANSFORM = "Transform must be one of {0}".format(transform_options)


class _DerivedMetric:
    def __init__(
        self,
        *,
        metric: Callable[..., Union[float, int]],
        transform: str,
        sample_param_names: List[str],
    ):
        if not callable(metric):
            raise ValueError(_METRIC_CALLABLE_ERROR)
        sig = inspect.signature(metric)

        for param_name in parameters_for_transforms:
            if param_name in sig.parameters:
                raise ValueError(_METHOD_ARG_ERROR.format(param_name))
        self._metric_fn = metric

        if transform not in transform_options:
            raise ValueError(_INVALID_TRANSFORM)
        self._transform = transform

        self._sample_param_names = []
        if sample_param_names is not None:
            self._sample_param_names = sample_param_names

    def __call__(
        self, y_true, y_pred, *, sensitive_features, **other_params
    ) -> Union[float, int]:
        sample_params = dict()
        params = dict()
        transform_parameters = dict()
        for k, v in other_params.items():
            if k in self._sample_param_names:
                sample_params[k] = v
            elif k in parameters_for_transforms:
                transform_parameters[k] = v
            else:
                params[k] = v

        dispatch_fn = functools.partial(self._metric_fn, **params)
        # Make sure there isn't a subsequent log message about
        # a nameless metric
        bound_fn_name = self._metric_fn.__name__
        for k, v in sorted(params.items()):
            bound_fn_name = bound_fn_name + "_" + k + "_" + str(v)
        dispatch_fn.__name__ = bound_fn_name

        all_metrics = MetricFrame(
            metrics=dispatch_fn,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            sample_params=sample_params,
        )

        if self._transform == "difference":
            result = all_metrics.difference(**transform_parameters)
        elif self._transform == "ratio":
            result = all_metrics.ratio(**transform_parameters)
        elif self._transform == "group_min":
            result = all_metrics.group_min()
        elif self._transform == "group_max":
            result = all_metrics.group_max()
        else:
            raise ValueError(_INVALID_TRANSFORM)

        return result


def make_derived_metric(
    *,
    metric: Callable[..., Union[float, int]],
    transform: str,
    sample_param_names: List[str] = ["sample_weight"],
) -> Callable[..., Union[float, int]]:
    """Create a scalar returning metric function based on aggregation of a disaggregated metric.

    Many higher order machine learning operations (such as hyperparameter tuning)
    make use of functions which return scalar metrics. We can create such a function
    for our disaggregated metrics with this function.

    This function takes a metric function, a string to specify the desired aggregation
    transform (matching the methods :meth:`MetricFrame.group_min`,
    :meth:`MetricFrame.group_max`, :meth:`MetricFrame.difference` and
    :meth:`MetricFrame.ratio`), and a list of
    parameter names to treat as sample parameters.

    The result is a callable object which has the same signature as the original
    function, with a :code:`sensitive_features=` parameter added.
    If the chosen aggregation transform accepts parameters (currently only
    :code:`method=` is supported), these can also be given when invoking the
    callable object.
    The result of this function is identical to
    creating a :class:`MetricFrame` object, and then calling the method specified
    by the :code:`transform=` argument (with the :code:`method=` argument, if
    required).

    See the :ref:`custom_fairness_metrics` section in the :ref:`user_guide` for more
    details.
    A :ref:`sample notebook <sphx_glr_auto_examples_plot_make_derived_metric.py>` is
    also available.

    Parameters
    ----------
    metric : callable
        The metric function from which the new function should be derived

    transform : str
        Selects the transformation aggregation the resultant function should use.
        The list of possible options is:
        ['difference', 'group_min', 'group_max', 'ratio'].

    sample_param_names : List[str]
        A list of parameters names of the underlying :code:`metric` which should
        be treated as sample parameters (i.e. the same leading dimension as the
        :code:`y_true` and :code:`y_pred` parameters). This defaults to a list with
        a single entry of :code:`sample_weight` (as used by many SciKit-Learn
        metrics). If :code:`None` or an empty list is supplied, then no parameters
        will be treated as sample parameters.

    Returns
    -------
    callable
        Function with the same signature as the :code:`metric` but with additional
        :code:`sensitive_features=` and :code:`method=` arguments, to enable the
        required computation
    """
    dm = _DerivedMetric(
        metric=metric, transform=transform, sample_param_names=sample_param_names
    )
    return dm
