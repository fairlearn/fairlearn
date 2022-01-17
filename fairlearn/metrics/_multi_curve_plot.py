# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ..utils._input_validation import (
    _validate_and_reformat_labels,
    _validate_and_reformat_labels_and_sf,
    check_consistent_length,
    _INPUT_DATA_FORMAT_ERROR_MESSAGE,
)
from ..postprocessing._plotting import _MATPLOTLIB_IMPORT_ERROR_MESSAGE
from ._make_derived_metric import _DerivedMetric
from typing import Callable, Union
from re import compile, match


def plot_model_comparison(
    *,
    x_axis_metric: Callable[..., Union[float, int]],
    y_axis_metric: Callable[..., Union[float, int]],
    y_true,
    y_preds,
    sensitive_features,
    ax=None,
    axis_labels=True,
    point_labels=True,
    group_by_name=False,
    color_gradient=False,
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

    y_preds : dict
        A dictionary mapping a model name (string) to its predictions (
        List, pandas.Series, numpy.ndarray, pandas.DataFrame)

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

    point_labels : bool
        If true, add the model name to the point.

    group_by_name : bool, str
        If true, derive groupings by common prefixes. In particular, model names
        that look like the regular expression :code:`\D.*. We use regular
        expression matching from the START of the string. If you want to
        use a different regular expression to group names, you can pass a
        RE pattern as well (for instance, :code:`group_by_name=r"\D.+"`).
        Tip: use cmap to change the color of different
        groups.

    color_gradient : bool
        If true, then a colormap gradient will be applied to
        the models. Colormaps can be supplied using the cmap kwarg.
        Colormap values are inferred from y_preds. If the
        model names are not all numbers, the index of the models become the
        colormap value.

    plot : bool
        If true, call pyplot.plot. In any case, return axis

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object that was drawn on.

    Notes
    -----
    To offer flexibility in plotting style, just as the
    underlying `matplotlib` provides,
    one has three options: 1) change the style of the returned Axes
    2) supply an Axes with your own style already applied
    3) supply matplotlib arguments as you normally
    would to `matplotlib.axes.Axes.scatter`

    In case no Axes object is supplied, axis labels are
    automatically inferred from their class name.
    """  # noqa: E501
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    if not isinstance(y_preds, dict):
        raise ValueError(
            _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                "y_preds", "dict", type(y_preds).__name__
            )
        )

    pattern = r"\D*"
    if not isinstance(group_by_name, bool):
        if isinstance(group_by_name, str):
            pattern = group_by_name
            group_by_name = True
        else:
            raise ValueError(
                "Keyword argument group_by_name must be a bool or a RE pattern"
            )

    if group_by_name and color_gradient:
        raise ValueError(
            "Can not set both keyword arguments "
            + "group_by_name and color_gradient to True"
        )

    # Input validation
    y_true, sensitive_features, _ = _validate_and_reformat_labels_and_sf(
        y_true, sensitive_features=sensitive_features
    )
    for key in y_preds:
        y_preds[key] = _validate_and_reformat_labels(y_preds[key])
    check_consistent_length(y_true, *list(y_preds.values()))

    # Calculate metrics
    # try-except structure because we expect: metric(y_true, y_pred, sensitive_attribute)
    # but we have as fallback: metric(y_true, y_pred)
    try:
        x = [
            x_axis_metric(
                y_true, y_preds[key], sensitive_features=sensitive_features
            )
            for key in y_preds
        ]
    except TypeError:
        x = [x_axis_metric(y_true, y_preds[key]) for key in y_preds]

    try:
        y = [
            y_axis_metric(
                y_true, y_preds[key], sensitive_features=sensitive_features
            )
            for key in y_preds
        ]
    except TypeError:
        y = [y_axis_metric(y_true, y_preds[key]) for key in y_preds]

    ax_supplied_ = ax is not None

    # Init ax
    if not ax_supplied_:
        ax = plt.axes()

    for (kwarg, name) in (
        (axis_labels, "axis_labels"),
        (point_labels, "point_labels"),
        (color_gradient, "color_gradient"),
        (plot, "plot"),
    ):
        if not isinstance(kwarg, bool):
            raise ValueError(
                _INPUT_DATA_FORMAT_ERROR_MESSAGE.format(
                    name, "boolean", type(kwarg).__name__
                )
            )

    # Add groupings
    groups = {}
    if group_by_name:
        pattern = compile(pattern)
        for i, key in enumerate(y_preds):
            # important: using match, so must be at start
            match_object = match(pattern, key)
            if not match_object:
                raise ValueError("can not match")
            group = match_object.group(0)
            remaining = key[match_object.end(0):]
            if groups.get(group, None) is None:
                groups[group] = {}
            # i here is the index in x and y
            groups[group][remaining] = i
        c = [0] * len(x)
        for i, group_key in enumerate(groups.keys()):
            group = groups[group_key]
            for item_key in group.keys():
                item = group[item_key]
                # item is the index of x and y
                # i is the index of the groups
                c[item] = i
        kwargs["c"] = c

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
    if point_labels:
        label = lambda i, t: ax.text(x[i], y[i], t)
        if group_by_name:
            for group in groups.keys():
                for item in groups[group].keys():
                    label(groups[group][item], item)
        else:
            for i, key in enumerate(y_preds):
                label(i, key)

    # Add color gradient
    if color_gradient:
        model_names_int_ = True
        for key in y_preds:
            if not isinstance(key, (int, float)):
                model_names_int_ = False

        if model_names_int_:
            model_names = [key for key in y_preds.keys()]
        else:
            model_names = [i for i, _ in enumerate(y_preds)]
        kwargs["c"] = model_names

    # Add to ax
    try:
        scatter = ax.scatter(
            x, y, **kwargs
        )  # Does it make sense to pass all other kwarg's?

        # NOTE: in order to support per-group things such as markers,
        # we must call scatter once per group. However, a nice
        # interface to support this will require some more thought.
        # However, for now, we are able to add a legend without calling
        # scatter once per group, but this requires the following call:
        if len(groups.keys()) > 0:
            ax.legend(
                handles=scatter.legend_elements(prop="colors")[0],
                labels=groups.keys(),
            )
    except AttributeError as e:
        # FIXME: Add some info?
        raise e

    if plot:
        plt.show()

    return ax
