# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import pandas as pd
import sklearn.metrics as skm

from fairlearn.metrics import MetricFunctionRequest, MetricFrame, MFv2


# Bring in some pre-prepared input arrays
from .data_for_test import y_t, y_p, g_1, g_2, g_3, g_4, s_w


class TestMetricFunctionRequest:
    def test_smoke(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1
        }
        df = pd.DataFrame.from_dict(data)

        target = MetricFunctionRequest(func=skm.recall_score)

        result = target.invoke(df)
        expected = skm.recall_score(y_t, y_p)

        assert result == expected


class TestMFv2:
    def test_overall_one_func(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1
        }
        df = pd.DataFrame.from_dict(data)

        wrapped_funcs = {
            'precision': MetricFunctionRequest(func=skm.precision_score)
        }
        target = MFv2(wrapped_funcs, df, ['group_1'])

        funcs = {
            'precision': skm.precision_score
        }
        expected = MetricFrame(metrics=funcs, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

        assert len(target.overall) == len(expected.overall) == 1
        assert target.overall['precision'] == expected.overall['precision']

    def test_overall_two_func(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1
        }
        df = pd.DataFrame.from_dict(data)

        wrapped_funcs = {
            'precision': MetricFunctionRequest(func=skm.precision_score),
            'accuracy': MetricFunctionRequest(func=skm.accuracy_score)
        }
        target = MFv2(wrapped_funcs, df, ['group_1'])

        funcs = {
            'precision': skm.precision_score,
            'accuracy': skm.accuracy_score
        }
        expected = MetricFrame(metrics=funcs, y_true=y_t, y_pred=y_p, sensitive_features=g_1)

        assert len(target.overall) == len(expected.overall) == 2
        assert target.overall['precision'] == expected.overall['precision']
        assert target.overall['accuracy'] == expected.overall['accuracy']
