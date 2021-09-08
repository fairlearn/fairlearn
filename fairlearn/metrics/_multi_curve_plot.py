from ..utils._input_validation import _validate_and_reformat_labels, _validate_and_reformat_labels_and_sf, check_consistent_length
from ..postprocessing._plotting import _MATPLOTLIB_IMPORT_ERROR_MESSAGE

def plot_model_comparison(
    *,
    x_axis_metric,
    y_axis_metric,
    y_true,
    y_preds,
    sensitive_features,
    show_plot,
    ax=None,
    **kwargs
    ):
    """Plot a model comparison

    Parameters
    ----------
    x_axis_metric : 
        something

    Returns
    -------
    ax
        todo
    
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
    x = [
        x_axis_metric(y_true, y_preds[key], sensitive_features=sensitive_features)
        for key in y_preds
    ]
    y = [
        y_axis_metric(y_true, y_preds[key], sensitive_features=sensitive_features)
        for key in y_preds
    ]

    # Plot
    if ax is None:
        ax = plt.axes()

    ax.scatter(x, y)
    
    if show_plot:
        plt.show()

    return ax
