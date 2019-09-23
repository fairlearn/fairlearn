# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

from . import MetricsResult


def true_positive_rate(y_actual, y_predict, group_id):
    data = pd.DataFrame({"y_actual": y_actual, "y_predict": y_predict, "group_id": group_id})

    data['true_positives'] = data.apply(lambda row: row.y_predict and row.y_actual, axis=1)

    counts = data.count()

    result = MetricsResult()
    result.metric = counts['true_postiives'] / counts['y_predict']

    return result
