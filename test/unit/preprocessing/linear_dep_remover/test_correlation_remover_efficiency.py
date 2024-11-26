from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fairlearn.datasets import fetch_adult
from fairlearn.metrics import (
    demographic_parity_difference,
    equal_opportunity_difference,
    equalized_odds_difference,
)
from fairlearn.preprocessing import CorrelationRemover


@dataclass(frozen=True)
class FairnessComparisonData:
    sensitive_feature: pd.Series
    y_true: pd.Series
    y_pred_with_mitigation: NDArray
    y_pred_without_mitigation: NDArray

    def bootstrap(self, seed: int) -> FairnessComparisonData:
        sample_indices = np.random.RandomState(seed=seed).choice(
            len(self.sensitive_feature), len(self.sensitive_feature), replace=True
        )
        sampled_sensitive_feature = self.sensitive_feature.iloc[sample_indices]
        sampled_y_true = self.y_true.iloc[sample_indices]
        sampled_y_pred_with_mitigation = self.y_pred_with_mitigation[sample_indices]
        sampled_y_pred_without_mitigation = self.y_pred_without_mitigation[sample_indices]

        return FairnessComparisonData(
            sensitive_feature=sampled_sensitive_feature,
            y_true=sampled_y_true,
            y_pred_with_mitigation=sampled_y_pred_with_mitigation,
            y_pred_without_mitigation=sampled_y_pred_without_mitigation,
        )


def _get_adult_fairness_comparison_data() -> FairnessComparisonData:
    features_to_keep = ["fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "age"]
    senstive_feature_id = "race_White"
    raw_data = fetch_adult().frame
    data, target = (
        raw_data[features_to_keep + ["race"]],
        raw_data["class"] == ">50K",
    )
    data = pd.get_dummies(data)[features_to_keep + [senstive_feature_id]]
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=0
    )

    cr = CorrelationRemover(sensitive_feature_ids=[senstive_feature_id])
    cr.fit(X_train)
    X_train_transformed = cr.transform(X_train)
    X_test_transformed = cr.transform(X_test)

    estimator_with_correlation = LogisticRegression(solver="liblinear")
    estimator_without_correlation = LogisticRegression(solver="liblinear")
    estimator_with_correlation.fit(X_train, y_train)
    estimator_without_correlation.fit(X_train_transformed, y_train)

    return FairnessComparisonData(
        sensitive_feature=X_test[senstive_feature_id],
        y_true=y_test,
        y_pred_without_mitigation=estimator_with_correlation.predict(X_test),
        y_pred_with_mitigation=estimator_without_correlation.predict(X_test_transformed),
    )


@pytest.mark.parametrize(
    "fairness_metrics",
    [[demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference]],
)
@pytest.mark.parametrize("fairness_comparison_data_loader", [_get_adult_fairness_comparison_data])
def test_correlation_remover_mitigates_bias(
    fairness_comparison_data_loader: Callable[[], FairnessComparisonData],
    fairness_metrics: list[Callable],
) -> None:
    data = fairness_comparison_data_loader()

    bootstrap_iterations = 30

    fairness_metrics_values_with_mitigation = np.zeros(
        (len(fairness_metrics), bootstrap_iterations)
    )
    fairness_metrics_values_without_mitigation = np.zeros(
        (len(fairness_metrics), bootstrap_iterations)
    )

    for iteration in range(bootstrap_iterations):
        sampled_data = data.bootstrap(seed=iteration)

        for i, metric in enumerate(fairness_metrics):
            metric_with_mitigation = metric(
                sampled_data.y_true,
                sampled_data.y_pred_with_mitigation,
                sensitive_features=sampled_data.sensitive_feature,
            )

            metric_without_mitigation = metric(
                sampled_data.y_true,
                sampled_data.y_pred_without_mitigation,
                sensitive_features=sampled_data.sensitive_feature,
            )

            fairness_metrics_values_with_mitigation[i, iteration] = metric_with_mitigation
            fairness_metrics_values_without_mitigation[i, iteration] = metric_without_mitigation

    for metric in range(len(fairness_metrics)):
        _, p_value = wilcoxon(
            fairness_metrics_values_without_mitigation[metric],
            fairness_metrics_values_with_mitigation[metric],
            alternative="greater",
        )

        assert p_value < 0.05
