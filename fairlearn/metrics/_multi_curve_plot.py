from ..utils._input_validation import _validate_and_reformat_labels, _validate_and_reformat_labels_and_sf, check_consistent_length
from ..postprocessing._plotting import _MATPLOTLIB_IMPORT_ERROR_MESSAGE
from ._make_derived_metric import _DerivedMetric
from typing import Callable, List, Union

def plot_model_comparison(
    *,
    x_axis_metric: Callable[..., Union[float, int]],
    y_axis_metric: Callable[..., Union[float, int]],
    y_true,
    y_preds,
    sensitive_features,
    show_plot: bool,
    ax=None,
    **kwargs
    ):
    """Plot a model comparison

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

    y_pred : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The predictions.

    sensitive_features : List, pandas.Series, dict of 1d arrays, numpy.ndarray, pandas.DataFrame, optional
        The sensitive features which should be used to create the subgroups.
        At least one sensitive feature must be provided.
        All names (whether on pandas objects or dictionary keys) must be strings.
        We also forbid DataFrames with column names of ``None``.
        For cases where no names are provided we generate names ``sensitive_feature_[n]``.
    
    show_plot : bool, optional
        When set to True, the generated pyplot will be drawn
    
    ax : matplotlib.axes.Axes, optional
        If supplied, the scatter plot is drawn on this Axes object.
        Else, a new figure with Axes is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object that was drawn on. If supplied
    
    Notes
    -----
    To offer flexibility in plotting style, just as the underlying `matplotlib` provides, 
    one has three options: 
        - (1) change the style of the returned Axes
        - (2) supply an Axes with your own style already applied
        - (3) supply matplotlib arguments as you normally would to `matplotlib.axes.Axes.scatter`

    In case no Axes object is supplied, axis labels are automatically inferred from their class name
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError(_MATPLOTLIB_IMPORT_ERROR_MESSAGE)

    # Input validation
    y_true, sensitive_features, _ = _validate_and_reformat_labels_and_sf(y_true, sensitive_features=sensitive_features)
    for key in y_preds:
        y_preds[key] = _validate_and_reformat_labels(y_preds[key])
    check_consistent_length(y_true, *list(y_preds.values()))

    # Calculate metrics
    # try-except structure because we expect: metric(y_true, y_pred, sensitive_attribute)
    # but we have as fallback: metric(y_true, y_pred)
    try:
        x = [
            x_axis_metric(y_true, y_preds[key], sensitive_features=sensitive_features)
            for key in y_preds
        ]
    except TypeError as e:
        x = [
            x_axis_metric(y_true, y_preds[key])
            for key in y_preds
        ]

    try:
        y = [
            y_axis_metric(y_true, y_preds[key], sensitive_features=sensitive_features)
            for key in y_preds
        ]
    except TypeError as e:
        y = [
            y_axis_metric(y_true, y_preds[key])
            for key in y_preds
        ]

    # Plot
    if ax is None:
        ax = plt.axes()

        # If no style was provided, we suggest these axis labels.
        # If an ax was provided, we rather not overwrite this.
        for f, m in [(ax.set_xlabel, x_axis_metric), (ax.set_ylabel, y_axis_metric)]:
            if hasattr(m, '__qualname__'): f(m.__qualname__)
            elif hasattr(m, '__name__'): f(m.__name__)
            elif isinstance(m, _DerivedMetric):
                f("%s, %s" % (m._metric_fn.__name__, m._transform))
            else: f(m.__repr__)
    
    try:
        ax.scatter(x, y, **kwargs) # Does it make sense to pass all other kwarg's?
    except AttributeError as e:
        raise TypeError("got an unexpected keyword argument")

    if show_plot:
        plt.show()

    return ax
