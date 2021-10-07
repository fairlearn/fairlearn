# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import pandas as pd
import sklearn.metrics as skm

from fairlearn.metrics import MetricFunctionRequest


# Bring in some pre-prepared input arrays
from .data_for_test import y_t, y_p, g_1, g_2, g_3, g_4, s_w


class TestMetricFunctionRequest:
    def test_smoke(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p
        }
        df = pd.DataFrame.from_dict(data)

        target = MetricFunctionRequest(func=skm.recall_score)

        result = target.invoke(df)
        expected = skm.recall_score(y_t, y_p)

        assert result == expected
