# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
import sklearn.metrics as skm

import time

from fairlearn.metrics import MetricFunctionRequest, MetricFrame, MFv2, selection_rate


# Bring in some pre-prepared input arrays
from .data_for_test import y_t, y_p, g_1, g_2, s_w
from .data_for_test import array_gen


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
        target = MFv2(metric_functions=wrapped_funcs, data=df, sensitive_features=['group_1'])

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
        target = MFv2(metric_functions=wrapped_funcs, data=df, sensitive_features=['group_1'])

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
        target = MFv2(metric_functions=wrapped_funcs, data=df, sensitive_features=['group_1'])

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
        target = MFv2(metric_functions=wrapped_funcs, data=df, sensitive_features=['group_1'])

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
        target = MFv2(metric_functions=wrapped_funcs, data=df,
                      sensitive_features=['group_1', 'group_2'])

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
        target = MFv2(metric_functions=wrapped_funcs, data=df,
                      sensitive_features=['group_1', 'group_2'])

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
            'my_sample_weight': s_w
        }
        df = pd.DataFrame.from_dict(data)

        wrapped_funcs = {
            'precision': MetricFunctionRequest(func=skm.precision_score),
            'prec_w': MetricFunctionRequest(func=skm.precision_score,
                                            args=['y_true', 'y_pred'],
                                            kwargs={'sample_weight': 'my_sample_weight'})
        }
        target = MFv2(metric_functions=wrapped_funcs, data=df, sensitive_features=['group_1'])

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
        assert target.overall['prec_w'] != \
            target.overall['precision'], "Weights should do something"

        assert target.by_group['precision']['aa'] == expected.by_group['precision']['aa']
        assert target.by_group['precision']['ba'] == expected.by_group['precision']['ba']
        assert target.by_group['prec_w']['aa'] == expected.by_group['prec_w']['aa']
        assert target.by_group['prec_w']['ba'] == expected.by_group['prec_w']['ba']

    def test_simple_selection_rate(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1,
            'group_2': g_2,
            'my_sample_weight': s_w
        }
        df = pd.DataFrame.from_dict(data)

        simple_selection_rate = MetricFunctionRequest(func=np.mean, args=['y_pred'])
        wrapped_funcs = {
            'ssr': simple_selection_rate,
            'sr': MetricFunctionRequest(func=selection_rate)
        }

        target = MFv2(metric_functions=wrapped_funcs, data=df, sensitive_features=['group_2'])

        assert target.overall['ssr'] == target.overall['sr']
        for g in np.unique(g_2):
            assert target.by_group['ssr'][g] == target.by_group['sr'][g]

    def test_1cf(self):
        data = {
            'y_true': y_t,
            'y_pred': y_p,
            'group_1': g_1,
            'group_2': g_2,
            'my_sample_weight': s_w
        }
        df = pd.DataFrame.from_dict(data)

        wrapped_funcs = {
            'precision': MetricFunctionRequest(func=skm.precision_score)
        }
        target = MFv2(metric_functions=wrapped_funcs,
                      data=df,
                      sensitive_features=['group_1'],
                      control_features=['group_2'])
        funcs = {
            'precision': skm.precision_score
        }
        expected = MetricFrame(
            metrics=funcs,
            y_true=y_t,
            y_pred=y_p,
            sensitive_features={'group_1': g_1},
            control_features={'group_2': g_2}
        )

        assert len(target.overall) == len(expected.overall) == 2
        assert target.overall['precision']['f'] == expected.overall['precision']['f']
        assert target.overall['precision']['g'] == expected.overall['precision']['g']

        assert len(target.by_group) == len(expected.by_group) == 4
        assert target.by_group['precision']['f']['aa'] == expected.by_group['precision']['f']['aa']
        assert target.by_group['precision']['f']['ba'] == expected.by_group['precision']['f']['ba']
        assert target.by_group['precision']['g']['aa'] == expected.by_group['precision']['g']['aa']
        assert target.by_group['precision']['g']['ba'] == expected.by_group['precision']['g']['ba']

    def test_perf(self):
        num_samples = 10000

        rng = np.random.default_rng(seed=2350135)

        metric_dict = {
            'recall': skm.recall_score,
            'prec': skm.precision_score,
            'acc': skm.accuracy_score
        }
        wrapped_dict = {
            'recall': MetricFunctionRequest(func=skm.recall_score),
            'prec': MetricFunctionRequest(func=skm.precision_score),
            'acc': MetricFunctionRequest(func=skm.accuracy_score)
        }

        # Generate 'true' and 'predicted' values
        y_true = rng.integers(low=0, high=1, endpoint=True, size=num_samples)
        y_pred = rng.integers(low=0, high=1, endpoint=True, size=num_samples)

        # Generate the conditional and sensitive features
        # Note that we have quite a few prime numbers here
        cf_1_groups = ['aa', 'aaa', 'aaaa']
        cf_2_groups = ['p', 'qq']
        sf_1_groups = ['A', 'B', 'C', 'D', 'E']
        sf_2_groups = ['H', 'L', 'M', 'P', 'Q', 'R', 'S']
        cf_1_arr = array_gen(cf_1_groups, 11, num_samples)
        cf_2_arr = array_gen(cf_2_groups, 13, num_samples)
        sf_1_arr = array_gen(sf_1_groups, 17, num_samples)
        sf_2_arr = array_gen(sf_2_groups, 19, num_samples)

        # Sanity check that all groups appear
        assert len(np.unique(cf_1_arr)) == len(cf_1_groups)
        assert len(np.unique(sf_1_arr)) == len(sf_1_groups)
        assert len(np.unique(cf_2_arr)) == len(cf_2_groups)
        assert len(np.unique(sf_2_arr)) == len(sf_2_groups)

        # Make some inputs
        sf_frame = pd.DataFrame(np.stack([sf_1_arr, sf_2_arr], axis=1), columns=["SF0", "SF1"])
        cf_array = pd.DataFrame(np.stack([cf_1_arr, cf_2_arr], axis=1), columns=['CF0', 'CF1'])

        # Create the target object

        begin = time.time()
        target = MetricFrame(metrics=metric_dict, y_true=y_true, y_pred=y_pred,
                             sensitive_features=sf_frame,
                             control_features=cf_array)
        end = time.time()

        delta_old = end - begin

        all_data = pd.concat(
            [
                pd.Series(name='y_true', data=y_true),
                pd.Series(name='y_pred', data=y_pred),
                sf_frame,
                cf_array
            ],
            axis=1
        )
        print(all_data.head())
        begin = time.time()
        other = MFv2(
            metric_functions=wrapped_dict,
            data=all_data,
            sensitive_features=["SF0", "SF1"],
            control_features=['CF0', 'CF1']
        )
        end = time.time()

        delta_new = end - begin

        print(f"delta_old: {delta_old}")
        print(f"delta_new: {delta_new}")

        for m in ['recall', 'prec', 'acc']:
            for cf_1 in cf_1_groups:
                for cf_2 in cf_2_groups:
                    for sf_1 in sf_1_groups:
                        for sf_2 in sf_2_groups:
                            assert target.by_group[m][cf_1][cf_2][sf_1][sf_2] ==\
                                other.by_group[m][cf_1][cf_2][sf_1][sf_2]
        assert delta_new < delta_old, "Check new version is faster"
