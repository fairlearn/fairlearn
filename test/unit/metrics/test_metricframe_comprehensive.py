# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

import fairlearn.metrics as metrics

from .data_for_test import array_gen


def test_3m_2sf_2cf():
    # This test goes through the entire structure
    # In theory this could replace almost all the other tests
    # However, it is large, complicated and programmatic

    # If this test fails, but the more focused tests do not,
    # then a focused test is probably missing

    num_samples = 10000

    rng = np.random.default_rng(seed=2350135)

    metric_dict = {
        "recall": skm.recall_score,
        "prec": skm.precision_score,
        "acc": skm.accuracy_score,
    }

    # Generate 'true' and 'predicted' values
    y_true = rng.integers(low=0, high=1, endpoint=True, size=num_samples)
    y_pred = rng.integers(low=0, high=1, endpoint=True, size=num_samples)

    # Generate the conditional and sensitive features
    # Note that we have quite a few prime numbers here
    cf_1_groups = ["aa", "aaa", "aaaa"]
    cf_2_groups = ["p", "qq"]
    sf_1_groups = ["A", "B", "C", "D", "E"]
    sf_2_groups = ["H", "L", "M", "P", "Q", "R", "S"]
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
    sf_frame = pd.DataFrame(
        np.stack([sf_1_arr, sf_2_arr], axis=1), columns=["SF0", "SF1"]
    )
    cf_array = np.stack([cf_1_arr, cf_2_arr], axis=1)

    # Create the target object
    target = metrics.MetricFrame(
        metrics=metric_dict,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sf_frame,
        control_features=cf_array,
    )

    # Check on the indices properties
    assert isinstance(target.control_levels, list)
    assert target.control_levels == ["control_feature_0", "control_feature_1"]
    assert isinstance(target.sensitive_levels, list)
    assert target.sensitive_levels == ["SF0", "SF1"]

    # Check everything
    for name, f in metric_dict.items():
        for cf_1 in cf_1_groups:
            mask_cf_1 = cf_1_arr == cf_1
            for cf_2 in cf_2_groups:
                mask_cf_2 = cf_2_arr == cf_2

                mask_cf = np.logical_and(mask_cf_1, mask_cf_2)
                assert sum(mask_cf) > 0

                # Check on overall
                overall = f(y_true[mask_cf], y_pred[mask_cf])
                assert target.overall[name][(cf_1, cf_2)] == overall

                metric_vals = []
                for sf_1 in sf_1_groups:
                    mask_sf_1 = sf_1_arr == sf_1
                    for sf_2 in sf_2_groups:
                        mask_sf_2 = sf_2_arr == sf_2

                        mask_sf = np.logical_and(mask_sf_1, mask_sf_2)
                        assert sum(mask_sf) > 0

                        mask = np.logical_and(mask_cf, mask_sf)
                        assert sum(mask) > 0

                        nxt = f(y_true[mask], y_pred[mask])
                        # Check by_group
                        assert target.by_group[name][(cf_1, cf_2, sf_1, sf_2)] == nxt
                        metric_vals.append(nxt)
                assert len(metric_vals) == len(sf_1_groups) * len(sf_2_groups)

                # Check on aggregates
                min_val = min(metric_vals)
                max_val = max(metric_vals)
                assert target.group_min()[name][(cf_1, cf_2)] == min_val
                assert target.group_max()[name][(cf_1, cf_2)] == max_val

                assert (
                    target.difference(method="between_groups")[name][(cf_1, cf_2)]
                    == max_val - min_val
                )
                assert (
                    target.ratio(method="between_groups")[name][(cf_1, cf_2)]
                    == min_val / max_val
                )

                diffs = [abs(x - overall) for x in metric_vals]
                assert target.difference(method="to_overall")[name][
                    (cf_1, cf_2)
                ] == max(diffs)

                # Ratio to overall needs a pytest.approx since we make sure
                # the ratio is less than 1 in a slightly different way
                ratios = [x / overall for x in metric_vals] + [
                    overall / x for x in metric_vals
                ]
                actual = target.ratio(method="to_overall")[name][(cf_1, cf_2)]
                assert actual == pytest.approx(min(ratios), rel=1e-10, abs=1e-16)

    # Finish off with a couple of very basic checks
    assert isinstance(target.overall, pd.DataFrame)
    assert target.overall.shape == (6, 3)
    assert isinstance(target.by_group, pd.DataFrame)
    assert target.by_group.shape == (210, 3)
