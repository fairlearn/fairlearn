# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import create_dashboard_dictionary
from fairlearn.metrics import create_group_metric_set


def test_create_dashboard_dictionary_smoke():
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
    groups = [1, 1, 1, 1, 2, 2, 2, 2]
    m_name = "my model"
    g_name = "my group"
    model_type = 'binary_classification'

    result = create_dashboard_dictionary(
        model_type,
        y_true,
        {m_name: y_pred},
        {g_name: groups})

    expected = create_group_metric_set(model_type,
                                       y_true,
                                       [y_pred],
                                       [groups],
                                       model_titles=[m_name],
                                       sensitive_feature_names=[g_name])

    assert result == expected
