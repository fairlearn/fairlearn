# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from fairlearn.metrics import create_dashboard_dictionary
from fairlearn.metrics._group_metric_set import create_group_metric_set


def test_create_dashboard_dictionary_smoke():
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0, 0, 1, 1]
    groups = [1, 1, 1, 1, 2, 2, 2, 2]
    m_name = "my model"
    g_name = "my group"
    prediction_type = 'binary_classification'

    actual = create_dashboard_dictionary(
        prediction_type,
        y_true,
        {m_name: y_pred},
        {g_name: groups})

    expected = create_group_metric_set(prediction_type,
                                       y_true,
                                       [y_pred],
                                       [groups],
                                       model_titles=[m_name],
                                       sensitive_feature_names=[g_name])

    assert expected == actual
    assert actual['schemaType'] == 'dashboardDictionary'
    assert actual['schemaVersion'] == 0


def test_create_dashboard_dictionary_multiple_models_multiple_sensitive_features():
    y_true = [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0]

    y_p1 = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1]
    y_p2 = [1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
    y_p3 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    m_1 = "M1"
    m_2 = "M2"
    m_3 = "M3"
    predictions = {m_1: y_p1, m_2: y_p2, m_3: y_p3}

    sf_1 = ['a', 'b', 'b', 'a', 'b', 'b', 'b', 'a', 'b', 'b', 'b']
    sf_2 = [4, 5, 6, 6, 5, 4, 4, 5, 5, 6, 6]
    n_1 = "sf 1"
    n_2 = "sf 2"
    sensitive_features = {n_1: sf_1, n_2: sf_2}

    prediction_type = 'binary_classification'

    actual = create_dashboard_dictionary(prediction_type,
                                         y_true,
                                         predictions,
                                         sensitive_features)

    expected = create_group_metric_set(prediction_type,
                                       y_true,
                                       [y_p1, y_p2, y_p3],
                                       [sf_1, sf_2],
                                       model_titles=[m_1, m_2, m_3],
                                       sensitive_feature_names=[n_1, n_2])

    assert expected == actual
    assert actual['schemaType'] == 'dashboardDictionary'
    assert actual['schemaVersion'] == 0
