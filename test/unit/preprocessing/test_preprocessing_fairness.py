from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

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
from fairlearn.preprocessing import CorrelationRemover, PrototypeRepresentationLearner

PreprocessingAlgorithm = Union[CorrelationRemover, PrototypeRepresentationLearner]


@dataclass(frozen=True)
class Dataset:
    sensitive_feature: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class FairnessComparisonData:
    sensitive_feature: pd.Series
    y_true: pd.Series
    y_pred_with_mitigation: NDArray
    y_pred_without_mitigation: NDArray

    def bootstrap(self, seed: int) -> FairnessComparisonData:
        sample_indices = np.random.default_rng(seed=seed).choice(
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


@pytest.fixture(scope="module")
def adult_dataset() -> Dataset:
    features_to_keep = ["fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "age"]
    sensitive_feature_id = "race_White"
    raw_data = fetch_adult().frame
    data, target = (
        raw_data[features_to_keep + ["race"]],
        raw_data["class"] == ">50K",
    )
    data = pd.get_dummies(data)[features_to_keep + [sensitive_feature_id]]
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=0
    )

    return Dataset(
        sensitive_feature=X_test[sensitive_feature_id],
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


@pytest.fixture(scope="module")
def dataset(request: pytest.FixtureRequest) -> Dataset:
    fixture_name = request.param
    return request.getfixturevalue(fixture_name)


@pytest.fixture(scope="module")
def mitigator(request: pytest.FixtureRequest) -> PreprocessingAlgorithm:
    return request.param


@pytest.fixture(scope="module")
def fairness_comparison_data(
    mitigator: PreprocessingAlgorithm, dataset: Dataset
) -> FairnessComparisonData:
    mitigator.fit(dataset.X_train)
    X_train_transformed = mitigator.transform(dataset.X_train)
    X_test_transformed = mitigator.transform(dataset.X_test)

    estimator_with_mitigation = LogisticRegression(solver="liblinear")
    estimator_without_mitigation = LogisticRegression(solver="liblinear")
    estimator_with_mitigation.fit(X_train_transformed, dataset.y_train)
    estimator_without_mitigation.fit(dataset.X_train, dataset.y_train)

    return FairnessComparisonData(
        sensitive_feature=dataset.sensitive_feature,
        y_true=dataset.y_test,
        y_pred_without_mitigation=estimator_without_mitigation.predict(dataset.X_test),
        y_pred_with_mitigation=estimator_with_mitigation.predict(X_test_transformed),
    )


@pytest.mark.parametrize(
    "fairness_metric",
    [demographic_parity_difference, equal_opportunity_difference, equalized_odds_difference],
)
@pytest.mark.parametrize(
    "mitigator",
    [
        CorrelationRemover(sensitive_feature_ids=["race_White"]),
        PrototypeRepresentationLearner(n_prototypes=4, max_iter=10),
    ],
    indirect=True,
)
@pytest.mark.parametrize("dataset", ["adult_dataset"], indirect=True)
def test_preprocessing_mitigates_bias(
    fairness_metric: Callable,
    fairness_comparison_data: FairnessComparisonData,
) -> None:
    bootstrap_iterations = 30

    fairness_metrics_values_with_mitigation = np.zeros(bootstrap_iterations)
    fairness_metrics_values_without_mitigation = np.zeros(bootstrap_iterations)

    for iteration in range(bootstrap_iterations):
        sampled_data = fairness_comparison_data.bootstrap(seed=iteration)

        metric_with_mitigation = fairness_metric(
            sampled_data.y_true,
            sampled_data.y_pred_with_mitigation,
            sensitive_features=sampled_data.sensitive_feature,
        )

        metric_without_mitigation = fairness_metric(
            sampled_data.y_true,
            sampled_data.y_pred_without_mitigation,
            sensitive_features=sampled_data.sensitive_feature,
        )

        fairness_metrics_values_with_mitigation[iteration] = metric_with_mitigation
        fairness_metrics_values_without_mitigation[iteration] = metric_without_mitigation

    _, p_value = wilcoxon(
        fairness_metrics_values_without_mitigation,
        fairness_metrics_values_with_mitigation,
        alternative="greater",
    )

    assert p_value < 0.05
