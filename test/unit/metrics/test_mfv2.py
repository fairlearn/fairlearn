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

    def test_by_group_one_func_one_group(self):
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
        expected = MetricFrame(metrics=funcs, y_true=y_t, y_pred=y_p,
                               sensitive_features={'group_1': g_1})

        assert len(target.by_group) == len(expected.by_group) == 2
        assert target.by_group['precision']['aa'] == expected.by_group['precision']['aa']
        assert target.by_group['precision']['ba'] == expected.by_group['precision']['ba']

    def test_by_group_two_func_one_group(self):
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
        expected = MetricFrame(metrics=funcs, y_true=y_t, y_pred=y_p,
                               sensitive_features={'group_1': g_1})

        assert len(target.by_group) == len(expected.by_group) == 2
        assert target.by_group['precision']['aa'] == expected.by_group['precision']['aa']
        assert target.by_group['precision']['ba'] == expected.by_group['precision']['ba']
        assert target.by_group['accuracy']['aa'] == expected.by_group['accuracy']['aa']
        assert target.by_group['accuracy']['ba'] == expected.by_group['accuracy']['ba']

    def test_by_group_one_func_two_group(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1,
            'group_2': g_2
        }
        df = pd.DataFrame.from_dict(data)

        wrapped_funcs = {
            'precision': MetricFunctionRequest(func=skm.precision_score)
        }
        target = MFv2(wrapped_funcs, df, ['group_1', 'group_2'])

        funcs = {
            'precision': skm.precision_score
        }
        expected = MetricFrame(metrics=funcs, y_true=y_t, y_pred=y_p,
                               sensitive_features={'group_1': g_1, 'group_2': g_2})

        assert len(target.by_group) == len(expected.by_group) == 4
        assert target.by_group['precision']['aa']['f'] == expected.by_group['precision']['aa']['f']
        assert target.by_group['precision']['ba']['f'] == expected.by_group['precision']['ba']['f']
        assert target.by_group['precision']['aa']['g'] == expected.by_group['precision']['aa']['g']
        assert target.by_group['precision']['ba']['g'] == expected.by_group['precision']['ba']['g']

    def test_by_group_two_func_two_group(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1,
            'group_2': g_2,
        }
        df = pd.DataFrame.from_dict(data)

        wrapped_funcs = {
            'precision': MetricFunctionRequest(func=skm.precision_score),
            'bal_acc': MetricFunctionRequest(func=skm.balanced_accuracy_score)
        }
        target = MFv2(wrapped_funcs, df, ['group_1', 'group_2'])

        funcs = {
            'precision': skm.precision_score,
            'bal_acc': skm.balanced_accuracy_score
        }
        expected = MetricFrame(metrics=funcs, y_true=y_t, y_pred=y_p,
                               sensitive_features={'group_1': g_1, 'group_2': g_2})

        assert len(target.by_group) == len(expected.by_group) == 4
        assert target.by_group['precision']['aa']['f'] == expected.by_group['precision']['aa']['f']
        assert target.by_group['precision']['ba']['f'] == expected.by_group['precision']['ba']['f']
        assert target.by_group['precision']['aa']['g'] == expected.by_group['precision']['aa']['g']
        assert target.by_group['precision']['ba']['g'] == expected.by_group['precision']['ba']['g']
        assert target.by_group['bal_acc']['aa']['f'] == expected.by_group['bal_acc']['aa']['f']
        assert target.by_group['bal_acc']['ba']['f'] == expected.by_group['bal_acc']['ba']['f']
        assert target.by_group['bal_acc']['aa']['g'] == expected.by_group['bal_acc']['aa']['g']
        assert target.by_group['bal_acc']['ba']['g'] == expected.by_group['bal_acc']['ba']['g']

    def test_sample_weights(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1,
            'group_2': g_2,
            'sample_weight': s_w
        }
        df = pd.DataFrame.from_dict(data)

        wrapped_funcs = {
            'precision': MetricFunctionRequest(func=skm.precision_score),
            'prec_w': MetricFunctionRequest(func=skm.precision_score, arguments=['y_true', 'y_pred', 'sample_weight'])
        }
        target = MFv2(wrapped_funcs, df, ['group_1'])

        funcs = {
            'precision': skm.precision_score,
            'prec_w': skm.precision_score
        }
        s_p = {'prec_w': {'sample_weight': s_w}}
        expected = MetricFrame(metrics=funcs, y_true=y_t, y_pred=y_p,
                               sensitive_features={'group_1': g_1},
                               sample_params=s_p)

        assert target.overall['precision'] == expected.overall['precision']
        assert target.overall['prec_w'] == expected.overall['prec_w']
        assert target.overall['prec_w'] != target.overall['precision'], "Weights should do something"

        assert target.by_group['precision']['aa'] == expected.by_group['precision']['aa']
        assert target.by_group['precision']['ba'] == expected.by_group['precision']['ba']
        assert target.by_group['prec_w']['aa'] == expected.by_group['prec_w']['aa']
        assert target.by_group['prec_w']['ba'] == expected.by_group['prec_w']['ba']
