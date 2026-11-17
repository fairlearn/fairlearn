# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame


@pytest.mark.parametrize("as_dict", [False, True])
@pytest.mark.parametrize("feature_kind", ["sensitive", "control"])
@pytest.mark.parametrize("feature_name", ["group", "y_true", "y_pred", "sample_weight"])
def test_feature_names_do_not_overwrite_metric_inputs(as_dict, feature_kind, feature_name):
    y_true = [0, 1, 1, 1]
    y_pred = [0, 0, 1, 1]
    weights = [1, 2, 3, 4]
    if feature_name == "sample_weight":
        feature_name = "accuracy_sample_weight" if as_dict else "None_sample_weight"
    groups = pd.Series([0, 0, 1, 1], name=feature_name)
    kwargs = {f"{feature_kind}_features": groups}
    if feature_kind == "control":
        kwargs["sensitive_features"] = ["all"] * 4
    sample_params = {"sample_weight": weights}
    frame = MetricFrame(
        metrics={"accuracy": accuracy_score} if as_dict else accuracy_score,
        y_true=y_true,
        y_pred=y_pred,
        sample_params={"accuracy": sample_params} if as_dict else sample_params,
        **kwargs,
    )

    assert frame.by_group.index.names[0] == feature_name
    np.testing.assert_allclose(np.asarray(frame.by_group).reshape(-1), [1 / 3, 1])
    if feature_kind == "control":
        assert frame.overall.index.name == feature_name
        np.testing.assert_allclose(np.asarray(frame.overall).reshape(-1), [1 / 3, 1])
    else:
        np.testing.assert_allclose(
            frame.overall, accuracy_score(y_true, y_pred, sample_weight=weights)
        )
        np.testing.assert_allclose(frame.difference(), 2 / 3)


def test_multiple_feature_name_collisions_with_bootstrap():
    y_true = [0, 1, 1, 1] * 4
    y_pred = [0, 0, 1, 1] * 4
    groups = pd.DataFrame({"y_true": [0, 0, 1, 1] * 4, "y_true_": ["all"] * 16})
    controls = pd.Series(["all"] * 16, name="y_pred")
    kwargs = {
        "metrics": accuracy_score,
        "y_true": y_true,
        "y_pred": y_pred,
        "n_boot": 5,
        "ci_quantiles": [0.25, 0.75],
        "random_state": 42,
    }
    reference = MetricFrame(
        sensitive_features=groups.rename(columns={"y_true": "group", "y_true_": "other"}),
        control_features=controls.rename("control"),
        **kwargs,
    )
    frame = MetricFrame(sensitive_features=groups, control_features=controls, **kwargs)
    assert frame.by_group.index.names == ["y_pred", "y_true", "y_true_"]
    np.testing.assert_allclose(frame.by_group, reference.by_group)
    np.testing.assert_allclose(frame.overall, reference.overall)
    np.testing.assert_allclose(frame.by_group_ci, reference.by_group_ci)
    np.testing.assert_allclose(frame.overall_ci, reference.overall_ci)
    np.testing.assert_allclose(frame.difference_ci(), reference.difference_ci())
