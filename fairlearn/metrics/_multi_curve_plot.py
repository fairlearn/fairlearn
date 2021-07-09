from .._input_validation import _validate_and_reformat_labels, _validate_and_reformat_labels_and_sf
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

    y_true, sensitive_features, _ = _validate_and_reformat_labels_and_sf(y_true, sensitive_features=sensitive_features)
    # CREATE PLOT OBJECT
    for key in y_preds:
        y_preds[key] = _validate_and_reformat_labels(y_preds[key])
        x_axis_metric(y_preds[key], y_preds)
        # COMPUTE X AND Y METRICS, MAP CURVE TO PLOT OBJECT
        # RANDOMIZE COLOURS?

    #RETURN PLOT OBJECT?




