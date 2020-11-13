# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import torch
from skorch import NeuralNetClassifier
from torch import nn, optim

from fairlearn.reductions import DemographicParity
from . import package_test_common as ptc


def create_model():
    # create model
    class BCECriterion(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.crit = nn.BCELoss(**kwargs)

        def forward(self, input, target):
            target = target.view(-1, 1)
            return self.crit(input, target.float())

    class ClassificationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(103, 12),
                nn.ReLU(),
                nn.Linear(12, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid())

        def forward(self, X, **kwargs):
            return self.model(X)

    class SampleWeightNeuralNet(NeuralNetClassifier):
        def __init__(self, *args, criterion__reduce=False, **kwargs):
            # make sure to set reduce=False in your criterion, since we need the loss
            # for each sample so that it can be weighted
            super().__init__(*args, criterion__reduce=criterion__reduce, **kwargs)

        def fit(self, X, y, sample_weight=None):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.to_numpy().astype('float32')
            if sample_weight is not None and isinstance(sample_weight, (pd.DataFrame, pd.Series)):
                sample_weight = sample_weight.to_numpy()

            sample_weight = sample_weight if sample_weight is not None else np.ones_like(y)
            X = {'X': X, 'sample_weight': sample_weight}
            return super().fit(X, y)

        def predict(self, X):
            if isinstance(X, (pd.DataFrame, pd.Series)):
                X = X.to_numpy().astype('float32')
            return super().predict(X)

        def get_loss(self, y_pred, y_true, X, *args, **kwargs):
            # override get_loss to use the sample_weight from X
            loss_unreduced = super().get_loss(y_pred, y_true, X, *args, **kwargs)
            sample_weight = X['sample_weight']
            # Need to put the sample weights on GPu
            loss_reduced = (sample_weight.to(loss_unreduced.device).unsqueeze(-1) * loss_unreduced).mean()
            return loss_reduced

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SampleWeightNeuralNet(
        ClassificationModel,
        max_epochs=10,
        optimizer=optim.Adam,
        lr=0.001,
        # No validation
        train_split=None,
        # Shuffle training data on each epoch
        iterator_train__shuffle=False,
        criterion=BCECriterion,
        device=device
    )
    return net


def test_expgrad_classification():
    estimator = create_model()
    disparity_moment = DemographicParity()

    ptc.run_expgrad_classification(estimator, disparity_moment)


def test_gridsearch_classification():
    estimator = create_model()
    disparity_moment = DemographicParity()

    ptc.run_gridsearch_classification(estimator, disparity_moment)


def test_thresholdoptimizer_classification():
    estimator = create_model()

    ptc.run_thresholdoptimizer_classification(estimator)
