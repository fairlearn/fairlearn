# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from ._group_metric_set import create_group_metric_set


def create_dashboard_dictionary(
        model_type,
        y_true,
        models,
        sensitive_features):
    """Create a dictionary compatible with the dashboard.

    :param model_type: Determines the metrics generated
    :type model_type: str
    :param y_true: The array of 'true' values
    :type y_true: list
    :param models: Dictionary of model names and y_pred values
    :type models: dict(str,list)
    :param sensitive_features: Dictionary of sensitive feature names and values
    :type sensitive_features: dict(str,list)
    """
    y_preds = []
    model_names = []
    for n, v in models.items():
        model_names.append(n)
        y_preds.append(v)

    s_f = []
    sf_names = []
    for n, v in sensitive_features.items():
        sf_names.append(n)
        s_f.append(v)

    return create_group_metric_set(model_type,
                                   y_true,
                                   y_preds,
                                   s_f,
                                   model_titles=model_names,
                                   sensitive_feature_names=sf_names)
