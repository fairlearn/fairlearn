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
from numpy import array
from sklearn.utils.validation import check_array
from numpy import zeros
import logging

logger = logging.getLogger(__name__)


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
    point_labels_position=(0, 0),
    legend=False,
    show_plot=False,
    **kwargs,
):
    r"""
    Create a scatter plot comparing multiple models along two metrics.

    A typical use case is when one of the metrics is a performance metric
    (e.g., balanced_accuracy) and the other is a fairness metric
    (e.g., false_negative_rate_difference).

    Parameters
    ----------
    y_preds : array-like, dict of array-like
        An array-like containing predictions per model. Hence, predictions of
        model :code:`i` should be in :code:`y_preds[i]`.

    y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The ground-truth labels (for classification) or target values (for regression).

    sensitive_features : List, pandas.Series, dict of 1d arrays, numpy.ndarray, pandas.DataFrame, optional
        Sensitive features for the fairness metrics (if a fairness metric is
        specified for the x-axis or the y-axis).

    x_axis_metric : Callable
        The metric function for the x-axis.
        The metric function must take `y_true`, `y_pred`, and optionally `sensitive_features`
        as arguments, and return a scalar value.

    y_axis_metric : Callable
        The metric function for the y-axis, similar to x_axis_metric.
        The metric function must take `y_true`, `y_pred`, and optionally `sensitive_features`
        as arguments, and return a scalar value.

    ax : matplotlib.axes.Axes, optional
        If supplied, the scatter plot is drawn on this Axes object.
        Else, a new figure with Axes is created.

    axis_labels : bool, list
        If true, add the names of x and y axis metrics. You can also pass a
        list of size two (or a two-tuple) of strings to use as axis labels instead.

    point_labels : bool, list
        If true, annotate text with inferred point labels. These labels are
        the keys of y_preds if y_preds is a dictionary, else simply the integers
        0...number of points - 1. You can specify point_labels as a list of
        labels as well.

    point_labels_position : list
        a list (or a two-tuple) containing precisely two numbers that define the
        offset of the point labels in the x and y direction respectively.
        The offset value is in data coordinates, not pixels.

    legend : bool
        If True, add a legend. Legend entries are created by passing the
        key word argument :code:`label` to calls to this function.
        If you want to customize the legend, you should manually call
        ax.legend (where ax is the Axes object) with your customization params

    show_plot : bool
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

    _, y_true, sensitive_features, _ = _validate_and_reformat_input(
        zeros((len(y_true), 1)),  # Dummy values for X
        y=y_true,
        sensitive_features=sensitive_features,
    )
    y_true = y_true.values
    sensitive_features = sensitive_features.values

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
        y_preds = array(y_preds)

    if len(y_preds.shape) == 1:
        y_preds = y_preds.reshape(1, -1)
    y_preds = check_array(y_preds)

    if not len(y_true) == y_preds.shape[1]:
        raise ValueError(
            _INCONSISTENT_ARRAY_LENGTH.format("y_true and the rows of y_preds")
        )

    if isinstance(axis_labels, (list, tuple)):
        if len(axis_labels) != 2:
            raise ValueError(
                "Key word argument axis_labels should be a list or tuple of two"
                " strings."
            )
    elif isinstance(axis_labels, bool):
        pass
    else:
        raise ValueError(
            _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                axis_labels,
                "boolean or tuple of two strings",
                type(axis_labels).__name__,
            )
        )

    for kwarg, name in (
        (legend, "legend"),
        (show_plot, "show_plot"),
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

    if (
        isinstance(point_labels_position, (list, tuple))
        and len(point_labels_position) == 2
    ):
        for item in point_labels_position:
            if not isinstance(item, (int, float)):
                raise ValueError(
                    "Key word argument point_labels_position is not a list or"
                    + " two-tuple containing only numbers."
                )
    else:
        raise ValueError("Key word argument point_labels_position is not a two-tuple.")

    # --- COMPUTE METRICS ---
    # try-except structure because we expect: metric(y_true, y_pred, sensitive_attribute)
    # but we have as fallback: metric(y_true, y_pred)
    try:
        x = array(
            [
                x_axis_metric(y_true, y_pred, sensitive_features=sensitive_features)
                for y_pred in y_preds
            ]
        )
    except TypeError:
        x = array([x_axis_metric(y_true, y_pred) for y_pred in y_preds])

    try:
        y = array(
            [
                y_axis_metric(y_true, y_pred, sensitive_features=sensitive_features)
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
            if hasattr(m, "__qualname__") and m.__qualname__ != "":
                name = m.__qualname__
            elif hasattr(m, "__name__") and m.__name__ != "":
                name = m.__name__
            elif isinstance(m, _DerivedMetric):
                name = f"{m._metric_fn.__name__}, {m._transform}"
            else:
                name = repr(m)
            f(name.replace("_", " "))
    elif isinstance(axis_labels, (tuple, list)):
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    # Add point labels
    # This could be nicer, but we rather not add a dependency on other packages.
    if point_labels is not None:
        x_offset, y_offset = point_labels_position
        for i, label in enumerate(point_labels):
            ax.text(x[i] + x_offset, y[i] + y_offset, label)

    # Add actual points
    ax.scatter(x, y, **kwargs)

    if legend:
        ax.legend()

    # User may want to draw in an ax but not plot it yet.
    if show_plot:
        plt.show()

    return ax
