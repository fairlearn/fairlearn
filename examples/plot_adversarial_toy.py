"""
.. _adversarial_Example_3:

==================================================
Adversarial Mitigation on a Synthetic Toy Problem
==================================================

This example reproduces the toy experiment from :footcite:t:`zhang2018mitigating`
with :class:`~fairlearn.adversarial.AdversarialFairnessClassifier`.

The data is small and fully synthetic, so we know exactly how the sensitive
feature leaks into the features and the label. That makes it easy to see what
the adversary changes: the predictor is a logistic regression, so we can read
off its weights before and after mitigation.
"""

# %%
# The data
# --------
#
# Each sample has a binary sensitive feature :math:`r`. A hidden variable
# :math:`v` is centered on :math:`r`, and both the observed feature :math:`u`
# and the variable :math:`w` that decides the label are noisy copies of
# :math:`v`:
#
# .. math::
#
#     r \sim \text{Bernoulli}(0.5), \quad
#     v \sim \mathcal{N}(r, 1), \quad
#     u \sim \mathcal{N}(v, 1), \quad
#     w \sim \mathcal{N}(v, 1), \quad
#     y = \mathbb{1}[w > 0].
#
# The model sees :math:`r` and :math:`u`. The label depends on :math:`r` only
# through :math:`v`, but because :math:`u` is a noisy measurement of :math:`v`,
# :math:`r` still helps predict :math:`y`, and samples with :math:`r = 1` get a
# positive label more often.

import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(123)
n = 100_000
r = rng.integers(0, 2, size=n)
v = rng.normal(loc=r, scale=1)
u = rng.normal(loc=v, scale=1)
w = rng.normal(loc=v, scale=1)

X = np.column_stack((r, u))
y = (w > 0).astype(int)

X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(
    X, y, r, test_size=0.2, random_state=42
)

# %%
# To compare models we look at accuracy and selection rate for each group,
# and at the demographic parity difference: the gap between the two groups'
# selection rates.

from sklearn.metrics import accuracy_score

from fairlearn.metrics import MetricFrame, demographic_parity_difference, selection_rate


def evaluate(y_pred):
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=r_test,
    )
    print(mf.by_group)
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=r_test)
    print(f"Overall accuracy: {mf.overall['accuracy']:.3f}")
    print(f"Demographic parity difference: {dpd:.3f}")


# %%
# Unmitigated model
# -----------------
#
# A plain logistic regression puts a large positive weight on :math:`r`,
# because it really is useful for predicting :math:`y`.

from sklearn.linear_model import LogisticRegression

baseline = LogisticRegression().fit(X_train, y_train)
print("weights (r, u):", baseline.coef_.ravel().round(2))
evaluate(baseline.predict(X_test))

# %%
# Adversarial mitigation
# ----------------------
#
# Now we train the same kind of model with an adversary. We build both networks
# ourselves in PyTorch so we can look at them afterwards. The predictor is a
# single linear layer followed by a sigmoid, which is a logistic regression. The
# adversary has the same shape: it gets the predictor's output and tries to
# recover :math:`r` from it.
#
# We use ``constraints="demographic_parity"``. With the default ``"auto"``
# losses, a binary label and a binary sensitive feature both use binary
# cross-entropy, so the adversary is itself a small logistic regression. If the
# sensitive feature were continuous, Fairlearn would use a squared-error loss
# for the adversary instead, and with ``predictor_loss`` and
# ``adversary_loss`` you can pass your own.
#
# ``alpha`` sets how hard the predictor works against the adversary. A small
# weight decay keeps the weights from drifting, as in the original experiment.
# Two passes over the training data are enough here.

import torch

from fairlearn.adversarial import AdversarialFairnessClassifier


def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)


torch.manual_seed(123)
predictor = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
adversary = torch.nn.Sequential(torch.nn.Linear(1, 1), torch.nn.Sigmoid())

mitigator = AdversarialFairnessClassifier(
    backend="torch",
    predictor_model=predictor,
    adversary_model=adversary,
    predictor_optimizer=optimizer,
    adversary_optimizer=optimizer,
    constraints="demographic_parity",
    alpha=1.0,
    epochs=2,
    batch_size=32,
    random_state=123,
)
mitigator.fit(X_train, y_train, sensitive_features=r_train)

# %%
# The mitigator trains the ``predictor`` we passed in, so we can read the
# weights of its linear layer the same way as before.

print("weights (r, u):", predictor[0].weight.detach().numpy().ravel().round(2))
evaluate(mitigator.predict(X_test))

# %%
# What changed
# ------------
#
# The weight on :math:`r` is now negative. Since :math:`u` is higher on average
# for :math:`r = 1`, a model that only used :math:`u` would still favor that
# group. To give both groups the same selection rate, the predictor uses
# :math:`r` to undo the part of :math:`u` that comes from group membership. The
# demographic parity difference drops close to zero, and we pay for it with
# some accuracy.
#
# Adversarial training is sensitive to its settings. If you change ``alpha``,
# the learning rate, or the number of steps, check the metrics again;
# :ref:`adversarial_Example_2` shows how to adjust them during training with
# callbacks.
#
# Keep in mind that this is a toy problem. It shows how the method works, not
# when to use it. Whether demographic parity is the right goal, and whether
# using the sensitive feature this way is acceptable, depends on the real
# deployment context and the harms involved. Synthetic data can't answer
# those questions.
#
# References
# ----------
#
# .. footbibliography::
