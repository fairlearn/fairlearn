# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate
from fairlearn.postprocessing import ThresholdOptimizer

fairness_metrics = {
    "accuracy": accuracy_score,
    "false_positive_rate": false_positive_rate,
    "positive_count": lambda true, pred: np.sum(true),
    "false_negative_rate": false_negative_rate,
    "negative_count": lambda true, pred: np.sum(1 - true),
    "balanced_accuracy": balanced_accuracy_score,
}


def test_threshold_optimizer_multiple_sensitive_features():
    # Create sensitive features so that the third column is the first two combined.
    # Also, the name a2 is long since that caused bug #728.
    # The bug caused the merged names to get cut off, resulting in multiple groups
    # getting merged internally. To avoid that this test case checks even internal
    # representations.
    X = pd.DataFrame(
        [
            [0, 4],
            [6, 2],
            [1, 3],
            [10, 5],
            [1, 7],
            [-2, 1],
            [3, 10],
            [14, 5],
            [1, 3],
            [1, 5],
            [1, 7],
            [-5, 9],
            [3, 13],
            [7, 1],
            [-8, 4],
            [9, 1],
        ]
    )
    y = pd.Series([0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0])
    a1 = "a"
    a2 = "a very very very very very very very long group name"
    a3 = "a group name with commas ,, in , it"
    a4 = "a group name with backslashes \\ in \\\\ it"
    A = pd.DataFrame(
        [
            [a1, a3, a1 + a3],
            [a1, a3, a1 + a3],
            [a2, a3, a2 + a3],
            [a2, a3, a2 + a3],
            [a2, a3, a2 + a3],
            [a2, a3, a2 + a3],
            [a2, a4, a2 + a4],
            [a2, a4, a2 + a4],
            [a2, a4, a2 + a4],
            [a2, a4, a2 + a4],
            [a2, a4, a2 + a4],
            [a2, a4, a2 + a4],
            [a2, a4, a2 + a4],
            [a2, a4, a2 + a4],
            [a1, a4, a1 + a4],
            [a1, a4, a1 + a4],
        ],
        columns=["SF1", "SF2", "SF1+2"],
    )

    estimator = LinearRegression()
    estimator.fit(X, y)

    postprocess_est_multi = ThresholdOptimizer(
        estimator=estimator,
        constraints="demographic_parity",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict",
    )
    postprocess_est_combined = ThresholdOptimizer(
        estimator=estimator,
        constraints="demographic_parity",
        objective="accuracy_score",
        prefit=True,
        predict_method="predict",
    )

    postprocess_est_multi.fit(X, y, sensitive_features=A.loc[:, ["SF1", "SF2"]])
    postprocess_est_combined.fit(X, y, sensitive_features=A.loc[:, "SF1+2"])

    X_test = pd.concat(
        [
            pd.DataFrame(
                [[5, 4], [7, 2], [0, 3], [1, 2], [-2, 9], [1, 1], [0, 5], [-3, 3]]
            ),
            X,
        ]
    )
    A_test = pd.concat(
        [
            pd.DataFrame(
                [
                    [a1, a3, a1 + a3],
                    [a1, a3, a1 + a3],
                    [a2, a3, a2 + a3],
                    [a2, a3, a2 + a3],
                    [a2, a4, a2 + a4],
                    [a2, a4, a2 + a4],
                    [a1, a4, a1 + a4],
                    [a1, a4, a1 + a4],
                ],
                columns=["SF1", "SF2", "SF1+2"],
            ),
            A,
        ]
    )
    y_test = pd.concat([pd.Series([0, 1, 0, 1, 0, 1, 0, 1]), y])

    y_pred_multi = postprocess_est_multi.predict(
        X_test, sensitive_features=A_test.loc[:, ["SF1", "SF2"]], random_state=1
    )
    y_pred_combined = postprocess_est_combined.predict(
        X_test, sensitive_features=A_test.loc[:, "SF1+2"], random_state=1
    )

    metricframe_multi = MetricFrame(
        metrics=fairness_metrics,
        y_true=y_test,
        y_pred=y_pred_multi,
        sensitive_features=A_test.loc[:, ["SF1", "SF2"]],
    )

    metricframe_combined = MetricFrame(
        metrics=fairness_metrics,
        y_true=y_test,
        y_pred=y_pred_combined,
        sensitive_features=A_test.loc[:, "SF1+2"],
    )

    # multi - names after escaping
    a3_escaped = a3.replace(",", "\\,")
    a4_escaped = a4.replace("\\", "\\\\")
    a13 = f"{a1},{a3_escaped}"
    a14 = f"{a1},{a4_escaped}"
    a23 = f"{a2},{a3_escaped}"
    a24 = f"{a2},{a4_escaped}"

    assert (metricframe_combined.overall == metricframe_multi.overall).all()

    assert (
        metricframe_combined.by_group.loc[a1 + a3]
        == metricframe_multi.by_group.loc[(a1, a3)]
    ).all()
    assert (
        metricframe_combined.by_group.loc[a2 + a3]
        == metricframe_multi.by_group.loc[(a2, a3)]
    ).all()
    assert (
        metricframe_combined.by_group.loc[a1 + a4]
        == metricframe_multi.by_group.loc[(a1, a4)]
    ).all()
    assert (
        metricframe_combined.by_group.loc[a2 + a4]
        == metricframe_multi.by_group.loc[(a2, a4)]
    ).all()

    # comparing string representations of interpolation dicts is sufficient
    assert str(
        postprocess_est_combined.interpolated_thresholder_.interpolation_dict[a1 + a3]
    ) == str(postprocess_est_multi.interpolated_thresholder_.interpolation_dict[a13])
    assert str(
        postprocess_est_combined.interpolated_thresholder_.interpolation_dict[a1 + a4]
    ) == str(postprocess_est_multi.interpolated_thresholder_.interpolation_dict[a14])
    assert str(
        postprocess_est_combined.interpolated_thresholder_.interpolation_dict[a2 + a3]
    ) == str(postprocess_est_multi.interpolated_thresholder_.interpolation_dict[a23])
    assert str(
        postprocess_est_combined.interpolated_thresholder_.interpolation_dict[a2 + a4]
    ) == str(postprocess_est_multi.interpolated_thresholder_.interpolation_dict[a24])
