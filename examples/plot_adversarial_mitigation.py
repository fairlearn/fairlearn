# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
================================================
Mitigating Fairness using Adversarial Mitigation
================================================
"""
# %%
# This notebook demonstrates our implementation of the technique *Mitigating*
# *Unwanted Biases with Adversarial Learning* as proposed by 
# `Zhang et al. 2018 <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_.
#
# In short, the authors take classic supervised learning setting in which
# a predictor neural network is trained, and extend it with an adversarial 
# network that aims to predict the sensitive feature. Then, they train the 
# predictor not only to minimize its own loss, but also minimize the predictive 
# ability of the adversarial.
#
# In short, we provide an implementation that supports:
# 
# - Any predictor neural network implemented in either PyTorch or Tensorflow2
# - Binary classification or TODO: single variable regression
# - One sensitive feature or TODO: multiple?
# - Two fairness objectives: Demographic parity or Equality of Odds
#
# This implementation follows closely the API of an `Estimator` in :code:`sklearn`
#
# Example 1: UCI Adult Dataset
# ============================
# Firstly, we load the dataset. For this example we choose the `sex` as the sensitive feature.
# We also preprocess the data such that every categorical column is translated to
# integer categories. Continuous-valued columns are left unchanged.

from fairlearn.datasets import fetch_adult
from sklearn.preprocessing import MinMaxScaler
from numpy import isnan

# Get dataset
data_bunch = fetch_adult()
X = data_bunch.data
y = data_bunch.target

# Remove rows with NaNs
non_NaN_rows = ~isnan(X).any(axis=1)
X = X[non_NaN_rows]
y = y[non_NaN_rows]

# Scale all columns to [0,1]
scaler = MinMaxScaler().fit(X) # scales to [0,1]
X = scaler.transform(X)

# Translate binary prediction from strings to 0 or 1
y = (y == ">50K").astype(float)

# Get sensitive feature
sensitive_feature = X[:, data_bunch.feature_names.index("sex")]


# %%
# Then, we define the predictor model using PyTorch. Note that we need to pass
# logits instead of sigmoidial values. TODO is this how u write it?

import torch
class FullyConnected(torch.nn.Module):
    def __init__(self, list_nodes):
        super(FullyConnected, self).__init__()
        layers = []
        for i in range(len(list_nodes)-1):
            layers.append(torch.nn.Linear(list_nodes[i], list_nodes[i+1]))
            layers.append(torch.nn.Sigmoid())
        layers.pop(-1)
        self.layers = torch.nn.ModuleList(layers)
        
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

predictor_model = FullyConnected(list_nodes = [14, 30, 20, 1])
# %%
# Now, we can use `fairlearn.adversarial.AdversarialMitigation` to train our
# predictor model on the UCI Adult dataset in a fair way. The specific fairness
# objective that we choose is for this example equality of odds, so we set
# :code:`objective = "EO"`.

from fairlearn.adversarial import AdversarialMitigation

mitigator = AdversarialMitigation(
        environment='torch',
        predictor_model = predictor_model,
        objective = "EO",
        cuda=True
)

# %% Then, we can fit the data to our model

mitigator.fit(X, y, sensitive_feature=sensitive_feature,
        epochs = 1000, batch_size=2**14, shuffle=True)

# %% Predict

y_pred = mitigator.predict(X)

# %% Evaluate

from fairlearn.metrics import equalized_odds_difference, MetricFrame, true_positive_rate, false_positive_rate, selection_rate
from sklearn.metrics import accuracy_score

EO_diff = equalized_odds_difference(y, y_pred, sensitive_features=sensitive_feature)
print("Equality of Odds difference: " + str(EO_diff))
mf = MetricFrame(
        metrics = {
                'accuracy': accuracy_score, 
                'selection_rate': selection_rate, 
                'tpr': true_positive_rate, 'fpr': false_positive_rate},
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature)

print(mf.by_group)

# %% Compare with normal predictor!

# %%
# From these results, we see that we have trained a ...