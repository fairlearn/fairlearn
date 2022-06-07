# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ..utils._input_validation import (
    _validate_and_reformat_input,
    _INCONSISTENT_ARRAY_LENGTH,
    _INPUT_DATA_FORMAT_ERROR_MESSAGE,
)
from ..postprocessing._plotting import _MATPLOTLIB_IMPORT_ERROR_MESSAGE
from ._make_derived_metric import _DerivedMetric
from typing import Callable, Union
from numpy import amax, amin, array
from sklearn.utils.validation import check_array
from numpy import zeros
import logging

logger = logging.getLogger(__name__)

context_state = None


def plot_model_comparison(
    *,
    y_preds,
    y_true=None,
    sensitive_features=None,
    x_axis_metric: Callable[..., Union[float, int]] = None,
    y_axis_metric: Callable[..., Union[float, int]] = None,
    ax=None,
    axis_labels=True,
    point_labels=False,
    legend=False,
    legend_kwargs={},
    plot=False,
    **kwargs,
):
    r"""
    Plot a model comparison.

    Parameters
    ----------
    y_preds : array-like, dict of array-like
        An array-like containing predictions per model. Hence, predictions of
        model :code:`i` should be in :code:`y_preds[i]`.

    y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The ground-truth labels (for classification) or target values (for regression).

    sensitive_features : List, pandas.Series, dict of 1d arrays, numpy.ndarray, pandas.DataFrame, optional
        The sensitive features which should be used to create the subgroups.
        At least one sensitive feature must be provided.
        All names (whether on pandas objects or dictionary keys) must be strings.
        We also forbid DataFrames with column names of ``None``.
        For cases where no names are provided
        we generate names ``sensitive_feature_[n]``.

    x_axis_metric : Callable
        The (aggregating) metric function for the x-axis
        The passed metric function must take `y_true, y_pred`, and optionally `sensitive_features`.
        If the metric is grouped, it must aggregate results. For instance, use
        `make_derived_metric(metric=balanced_accuracy_score, transform='group_min')`
        to aggregate the `balanced_accuracy_score`.

    y_axis_metric : Callable
        The (aggregating) metric function for the y-axis, similar to x_axis_metric.
        The passed metric function must take `y_true, y_pred`, and optionally `sensitive_features`.
        If the metric is grouped, it must aggregate results.

    ax : matplotlib.axes.Axes, optional
        If supplied, the scatter plot is drawn on this Axes object.
        Else, a new figure with Axes is created.

    axis_labels : bool, tuple
        If true, add the names of x and y axis metrics. You can also pass a
        two-tuple of strings to use as axis labels instead.

    point_labels : bool, list
        If true, annotate text with inferred point labels. These labels are
        the keys of y_preds if y_preds is a dictionary, else simply the integers
        0...number of points - 1. You can specify point_labels as a list of
        labels as well.

    legend : bool
        If True, add a legend. Legend entries are created by passing the
        key word argument :code:`label` to calls to this function.

    legend_kwargs : dict
        Keyword arguments passed to :py:func:`Axes.legend`. For instance,
        :code:`legend_kwargs={'bbox_to_anchor': (1.03,1), 'loc': 'upper left'}`
        will create a legend to the right of the Axes (subfigure), or
        :code:`legend_kwargs={'ncol': 2}` will create two columns in the legend.

    plot : bool
        If true, call pyplot.plot.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object that was drawn on.

    Notes
    -----
    To offer flexibility in stylistic features besides the aforementioned
    API options, one has at least three options: 1) supply matplotlib arguments
    to :code:`plot_model_comparison` as you normally
    would to :code:`matplotlib.axes.Axes.scatter`
    2) change the style of the returned Axes
    3) supply an Axes with your own style already applied

    In case no Axes object is supplied, axis labels are
    automatically inferred from their class name.
    """  # noqa: E501
    # --- CHECK DEPENDENCY ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    # --- SET OR LOAD CONTEXT STATE ---
    global context_state
    context_kws = (y_true, sensitive_features, x_axis_metric, y_axis_metric)
    if all(kw is None for kw in context_kws):
        if context_state is None:
            raise ValueError(
                "Must provide the following key word arguments on first call: "
                + "y_true, sensitive_features, x_axis_metric, y_axis_metric."
            )
        (
            y_true,
            sensitive_features,
            x_axis_metric,
            y_axis_metric,
        ) = context_state
    elif any(kw is None for kw in context_kws):
        raise ValueError(
            "Either provide all or none of the following key word arguments: "
            + "y_true, sensitive_features, x_axis_metric, y_axis_metric."
        )
    else:
        _, y_true, sensitive_features, _ = _validate_and_reformat_input(
            zeros((len(y_true), 1)),  # Dummy values for X
            y=y_true,
            sensitive_features=sensitive_features,
        )
        y_true = y_true.values
        sensitive_features = sensitive_features.values

        context_state = (
            y_true,
            sensitive_features,
            x_axis_metric,
            y_axis_metric,
        )

    # --- VALIDATE INPUT ---
    if isinstance(y_preds, dict):
        inferred_point_labels = []
        y_preds_list = []
        for key, value in y_preds.items():
            inferred_point_labels.append(key)
            y_preds_list.append(value)
        y_preds = array(y_preds_list)
    else:
        inferred_point_labels = range(len(y_preds))

    if len(y_preds.shape) == 1:
        y_preds = y_preds.reshape(1, -1)
    y_preds = check_array(y_preds)

    if not len(y_true) == y_preds.shape[1]:
        raise ValueError(
            _INCONSISTENT_ARRAY_LENGTH.format("y_true and the rows of y_preds")
        )

    if isinstance(axis_labels, tuple):
        if len(axis_labels) != 2:
            raise ValueError(
                "Key word argument axis_labels should be a tuple of two strings."
            )
    elif isinstance(axis_labels, bool):
        pass
    else:
        raise ValueError(
            _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                axis_labels,
                "boolean or tuple of two strings",
                type(kwarg).__name__,
            )
        )

    for (kwarg, name) in (
        (legend, "legend"),
        (plot, "plot"),
    ):
        if not isinstance(kwarg, bool):
            raise ValueError(
                _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                    name, "boolean", type(kwarg).__name__
                )
            )

    if isinstance(point_labels, list):
        if not len(point_labels) == y_preds.shape[0]:
            raise ValueError(
                _INCONSISTENT_ARRAY_LENGTH.format("point_labels and y_preds")
            )
    elif isinstance(point_labels, bool):
        if point_labels:
            point_labels = inferred_point_labels
        else:
            point_labels = None
    else:
        raise ValueError(
            _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                name, "boolean or list of labels", type(kwarg).__name__
            )
        )

    if not isinstance(legend_kwargs, dict):
        raise ValueError(
            _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                "legend_kwargs", "dict", type(legend_kwargs).__name__
            )
        )

    # --- COMPUTE METRICS ---
    # try-except structure because we expect: metric(y_true, y_pred, sensitive_attribute)
    # but we have as fallback: metric(y_true, y_pred)
    try:
        x = array(
            [
                x_axis_metric(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
                for y_pred in y_preds
            ]
        )
    except TypeError:
        x = array([x_axis_metric(y_true, y_pred) for y_pred in y_preds])

    try:
        y = array(
            [
                y_axis_metric(
                    y_true, y_pred, sensitive_features=sensitive_features
                )
                for y_pred in y_preds
            ]
        )
    except TypeError:
        y = array([y_axis_metric(y_true, y_pred) for y_pred in y_preds])

    # --- PLOTTING ---
    ax_supplied_ = ax is not None

    # Init ax
    if not ax_supplied_:
        logger.warning(
            "No matplotlib.Axes object was provided to draw on, so we create a new one"
        )
        ax = plt.axes()

    # Add axis labels
    if isinstance(axis_labels, bool):
        for f, m in (
            (ax.set_xlabel, x_axis_metric),
            (ax.set_ylabel, y_axis_metric),
        ):
            if hasattr(m, "__qualname__"):
                name = m.__qualname__
            elif hasattr(m, "__name__"):
                name = m.__name__
            elif isinstance(m, _DerivedMetric):
                name = f"{m._metric_fn.__name__}, {m._transform}"
            else:
                name = m.__repr__
            f(name.replace("_", " "))
    elif isinstance(axis_labels, tuple):
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    # Add point labels
    # This could be nicer, but we rather not add a dependency on other packages.
    if point_labels is not None:
        for i, label in enumerate(point_labels):
            ax.text(x[i], y[i], label)

    # Add actual points
    ax.scatter(x, y, **kwargs)

    # FIXME: @hildeweerts Do we need to set xlim/ylim separately?
    # NOTE: If so, I think matplotlib automatically has 5% margin
    # Also, we need to store extremas of previous calls to scatter
    # x_max, x_min, y_max, y_min = amax(x), amin(x), amax(y), amin(y)
    # x_margin, y_margin = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
    # ax.set_xlim(left=x_min - x_margin, right=x_max + x_margin)
    # ax.set_ylim(bottom=y_min - y_margin, top=y_max + y_margin)

    if legend:
        ax.legend(**legend_kwargs)

    # User may want to draw in an ax but not plot it yet.
    if plot:
        plt.show()

    return ax
