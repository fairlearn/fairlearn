# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from fairlearn.adversarial import AdversarialFairnessClassifier
from test._sklearn_compat import parametrize_with_checks


def _estimator(learning_rate=0.001):
    return AdversarialFairnessClassifier(
        predictor_model=object(),
        adversary_model=object(),
        learning_rate=learning_rate,
    )


def _estimator_id(estimator):
    ids = parametrize_with_checks([estimator]).mark.kwargs["ids"]
    return ids(estimator)


def test_estimator_check_ids_are_independent_of_object_addresses():
    # Keep both estimators alive so the allocator cannot reuse their addresses.
    first, second = _estimator(), _estimator()
    assert first.predictor_model is not second.predictor_model
    assert _estimator_id(first) == _estimator_id(second)


def test_estimator_check_ids_preserve_hyperparameters():
    first, second = _estimator(0.01), _estimator(0.02)
    assert _estimator_id(first) != _estimator_id(second)
