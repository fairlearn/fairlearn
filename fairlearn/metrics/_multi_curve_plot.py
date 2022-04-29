# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ..utils._input_validation import (
    _validate_and_reformat_labels_and_sf,
    _INCONSISTENT_ARRAY_LENGTH,
    _INPUT_DATA_FORMAT_ERROR_MESSAGE,
)
from ..postprocessing._plotting import _MATPLOTLIB_IMPORT_ERROR_MESSAGE
from ._make_derived_metric import _DerivedMetric
from typing import Callable, Union
from numpy import amax, amin, array
from sklearn.utils.validation import check_array


def plot_model_comparison(
    *,
    x_axis_metric: Callable[..., Union[float, int]],
    y_axis_metric: Callable[..., Union[float, int]],
    y_true,
    y_preds,
    sensitive_features,
    ax=None,
    axis_labels=True,
    point_labels=None,
    model_kwargs=None,
    legend=False,
    legend_kwargs={},
    plot=True,
    **kwargs,
):
    r"""
    Plot a model comparison.

    Parameters
    ----------
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

    y_true : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The ground-truth labels (for classification) or target values (for regression).

    y_preds : array-like
        An array-like containing predictions per model. Hence, predictions of
        model :code:`i` should be in :code:`y_preds[i]`.

    sensitive_features : List, pandas.Series, dict of 1d arrays, numpy.ndarray, pandas.DataFrame, optional
        The sensitive features which should be used to create the subgroups.
        At least one sensitive feature must be provided.
        All names (whether on pandas objects or dictionary keys) must be strings.
        We also forbid DataFrames with column names of ``None``.
        For cases where no names are provided
        we generate names ``sensitive_feature_[n]``.

    ax : matplotlib.axes.Axes, optional
        If supplied, the scatter plot is drawn on this Axes object.
        Else, a new figure with Axes is created.

    axis_labels : bool
        If true, add the names of x and y axis metrics

    point_labels : list
        Add textual label :code:`point_labels[i]` to the position
        of model :code:`i`.

    model_kwargs : list[dict], optional
        For a model at index :code:`i` (same order as in :code:`y_preds`),
        :code:`model_kwargs[i]` are passed along to matplotlib's scatter
        plotting. If models have the same `model_kwargs`, they are passed
        in one call to :py:func:`Axes.scatter`, so they become one entry in
        the legend.

    legend : bool
        If True, add a legend. Must set
        :code:`model_kwargs[i]['label']` for every prediction :code:`i`.

    legend_kwargs : dict
        Keyword arguments passed to :py:func:`Axes.legend`. For instance,
        :code:`legend_kwargs={'bbox_to_anchor': (1.03,1), 'loc': 'upper left'}`
        will create a legend to the right of the Axes (subfigure), or
        :code:`legend_kwargs={'ncol': 2}` will create two columns in the legend.

    plot : bool
        If true, call pyplot.plot. In any case, return axis

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

    # --- VALIDATE INPUT ---
    y_preds = check_array(y_preds)

    # Input validation
    y_true, sensitive_features, _ = _validate_and_reformat_labels_and_sf(
        y_true, sensitive_features=sensitive_features
    )

    if not len(y_true) == y_preds.shape[1]:
        raise ValueError(
            _INCONSISTENT_ARRAY_LENGTH.format("y_true and the rows of y_preds")
        )

    for (kwarg, name) in (
        (axis_labels, "axis_labels"),
        (legend, "legend"),
        (plot, "plot"),
    ):
        if not isinstance(kwarg, bool):
            raise ValueError(
                _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                    name, "boolean", type(kwarg).__name__
                )
            )

    for (kwarg, name) in (
        (point_labels, "point_labels"),
        (model_kwargs, "model_kwargs"),
    ):
        if kwarg is not None:
            if not isinstance(kwarg, list):
                raise ValueError(
                    _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                        name, "list", type(kwarg).__name__
                    )
                )
            if not len(kwarg) == y_preds.shape[0]:
                raise ValueError(
                    _INCONSISTENT_ARRAY_LENGTH.format(f"{name} and y_preds")
                )

    if model_kwargs is not None:
        for item in model_kwargs:
            if not isinstance(item, dict):
                raise ValueError(
                    _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                        "items of model_kwargs", "dict", type(item).__name__
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
        ax = plt.axes()

    # Add axis labels
    if axis_labels:
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

    # Add point labels
    # This could be nicer, but we rather not add a dependency on other packages.
    if point_labels is not None:
        for i, label in enumerate(point_labels):
            ax.text(x[i], y[i], label)

    # Add actual points
    try:
        if model_kwargs is None:
            ax.scatter(x, y, **kwargs)
        else:
            # Find the models with the same model_kwargs, and pass them into
            # a single call to scatter.
            is_first = [True for _ in model_kwargs]
            equivalence = [[i] for i in range(len(model_kwargs))]
            for i, mkws1 in enumerate(model_kwargs):
                if not is_first[i]:
                    continue
                for j, mkws2 in enumerate(model_kwargs[i + 1:], start=i + 1):
                    if mkws1 == mkws2:
                        is_first[j] = False
                        equivalence[i].append(j)

            # FIXME: @hildeweerts Do we need to set xlim/ylim separately?
            # NOTE: If so, I think matplotlib automatically has 5% margin
            x_max, x_min, y_max, y_min = amax(x), amin(x), amax(y), amin(y)
            x_margin, y_margin = (x_max - x_min) * 0.05, (y_max - y_min) * 0.05
            ax.set_xlim(left=x_min - x_margin, right=x_max + x_margin)
            ax.set_ylim(bottom=y_min - y_margin, top=y_max + y_margin)

            for i, this_model_kwargs in enumerate(model_kwargs):
                if is_first[i]:
                    index = equivalence[i]
                    kws = kwargs.copy()
                    kws.update(this_model_kwargs)
                    ax.scatter(x[index], y[index], **kws)
        if legend:
            ax.legend(**legend_kwargs)
    except AttributeError as e:
        # FIXME: Add some info, as this is probably because of wrong kwargs.
        raise e

    # User may want to draw in an ax but not plot it yet.
    if plot:
        plt.show()

    return ax
