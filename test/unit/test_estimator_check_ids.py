# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.adversarial import AdversarialFairnessClassifier
from test._sklearn_compat import parametrize_with_checks


def _estimator_id(learning_rate=0.001):
    estimator = AdversarialFairnessClassifier(
        predictor_model=object(),
        adversary_model=object(),
        learning_rate=learning_rate,
    )
    ids = parametrize_with_checks([estimator]).mark.kwargs["ids"]
    return ids(estimator)


def test_estimator_check_ids_are_independent_of_object_addresses():
    assert _estimator_id() == _estimator_id()


def test_estimator_check_ids_preserve_hyperparameters():
    assert _estimator_id(0.01) != _estimator_id(0.02)
