# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from fairlearn.adversarial import AdversarialFairnessClassifier

torch = pytest.importorskip("torch")


def test_supplied_sigmoid_predictor_keeps_probability_loss_and_prediction_contract():
    class SigmoidPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)
            self.activation = torch.nn.Sigmoid()

        def forward(self, X):
            return self.activation(self.linear(X))

    X = np.array([[-1.0, 0.0], [1.0, 0.0], [-0.5, 0.0], [0.5, 0.0]])
    y = np.array([0, 1, 0, 1])
    sensitive = np.array([0, 1, 1, 0])
    predictor = SigmoidPredictor()
    with torch.no_grad():
        predictor.linear.weight[:] = torch.tensor([[0.4, 0.0]])
        predictor.linear.bias[:] = 0
    mitigator = AdversarialFairnessClassifier(
        backend="torch",
        predictor_model=predictor,
        epochs=1,
        batch_size=-1,
        progress_updates=0,
        random_state=42,
    ).fit(X, y, sensitive_features=sensitive)

    assert isinstance(mitigator.backendEngine_.predictor_loss, torch.nn.BCELoss)
    probabilities = mitigator.backendEngine_.evaluate(X)
    np.testing.assert_array_equal(
        mitigator.predict(X)[:, np.newaxis], (probabilities >= 0.5).astype(float)
    )


def test_supplied_softmax_predictor_uses_probability_cross_entropy():
    class SoftmaxPredictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 3)
            self.activation = torch.nn.Softmax(dim=1)

        def forward(self, X):
            return self.activation(self.linear(X))

    X = np.array([[-1.0, 0.0], [1.0, 0.0], [-0.5, 0.0], [0.5, 0.0]])
    y = np.array([0, 1, 2, 1])
    sensitive = np.array([0, 1, 1, 0])
    predictor = SoftmaxPredictor()
    mitigator = AdversarialFairnessClassifier(
        backend="torch",
        predictor_model=predictor,
        epochs=1,
        batch_size=-1,
        progress_updates=0,
        random_state=42,
    ).fit(X, y, sensitive_features=sensitive)

    target = torch.tensor([[1.0, 0.0, 0.0]])
    probability = torch.tensor([[0.8, 0.1, 0.1]])
    loss = mitigator.backendEngine_.predictor_loss(probability, target)
    assert loss.item() == pytest.approx(-np.log(0.8))
