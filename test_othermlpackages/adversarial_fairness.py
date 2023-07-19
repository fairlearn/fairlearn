# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""All example code from "docs/user_guide/adversarial.rst"."""

from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import torch
from numpy import mean, number

import fairlearn.utils._compatibility as compat
from fairlearn.datasets import fetch_adult
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
)
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from fairlearn.adversarial import AdversarialFairnessClassifier

# Global variables of test_examples()
schedulers = []
step = 1


def test_examples():
    # EXAMPLE 1
    X, y = fetch_adult(return_X_y=True)
    pos_label = y[0]

    z = X["sex"]  # In this example, we consider 'sex' the sensitive feature.

    ct = make_column_transformer(
        (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("normalizer", StandardScaler()),
                ]
            ),
            make_column_selector(dtype_include=number),
        ),
        (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(drop="if_binary", **compat._SPARSE_OUTPUT_FALSE),
                    ),
                ]
            ),
            make_column_selector(dtype_include="category"),
        ),
    )

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, y, z, test_size=0.2, random_state=12345, stratify=y
    )

    X_prep_train = ct.fit_transform(X_train)  # Only fit on training data!
    X_prep_test = ct.transform(X_test)

    mitigator = AdversarialFairnessClassifier(
        backend="torch",
        predictor_model=[50, "leaky_relu"],
        adversary_model=[3, "leaky_relu"],
        batch_size=2**8,
        progress_updates=0.5,
        random_state=123,
    )

    mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

    predictions = mitigator.predict(X_prep_test)

    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=Y_test == pos_label,
        y_pred=predictions == pos_label,
        sensitive_features=Z_test,
    )

    RESULT1 = mf.by_group

    # EXAMPLE 2
    class PredictorModel(torch.nn.Module):
        def __init__(self):
            super(PredictorModel, self).__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(X_prep_train.shape[1], 200),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(200, 1),
                torch.nn.Sigmoid(),
            )

        def forward(self, x):
            return self.layers(x)

    predictor_model = PredictorModel()

    def validate(mitigator):
        predictions = mitigator.predict(X_prep_test)
        dp_diff = demographic_parity_difference(
            Y_test == pos_label,
            predictions == pos_label,
            sensitive_features=Z_test,
        )
        accuracy = mean(predictions.values == Y_test.values)
        selection_rate = mean(predictions == pos_label)
        return dp_diff, accuracy, selection_rate

    def optimizer_constructor(model):
        global schedulers
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        schedulers.append(
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
        )
        return optimizer

    from math import sqrt

    def callbacks(model, *args):
        global step
        global schedulers
        step += 1
        # Update hyperparameters
        model.alpha = 0.3 * sqrt(step // 1)
        for scheduler in schedulers:
            scheduler.step()
        # Validate (and early stopping) every 50 steps
        if step % 50 == 0:
            dp_diff, accuracy, selection_rate = validate(model)
            # Early stopping condition:
            # Good accuracy + low dp_diff + no mode collapse
            if (
                dp_diff < 0.03
                and accuracy > 0.8
                and selection_rate > 0.01
                and selection_rate < 0.99
            ):
                return True

    mitigator = AdversarialFairnessClassifier(
        predictor_model=predictor_model,
        adversary_model=[3, "leaky_relu"],
        predictor_optimizer=optimizer_constructor,
        adversary_optimizer=optimizer_constructor,
        epochs=10,
        batch_size=2**7,
        shuffle=True,
        callbacks=callbacks,
        random_state=123,
    )

    mitigator.fit(X_prep_train, Y_train, sensitive_features=Z_train)

    RESULT2a = validate(mitigator)

    predictions = mitigator.predict(X_prep_test)

    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=Y_test == pos_label,
        y_pred=predictions == pos_label,
        sensitive_features=Z_test,
    )

    RESULT2b = mf.by_group

    pipeline = Pipeline(
        [
            ("preprocessor", ct),
            (
                "classifier",
                AdversarialFairnessClassifier(
                    backend="torch",
                    predictor_model=[50, "leaky_relu"],
                    adversary_model=[3, "leaky_relu"],
                    batch_size=2**8,
                    random_state=123,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, Y_train, classifier__sensitive_features=Z_train)

    predictions = pipeline.predict(X_test)

    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
        y_true=Y_test == pos_label,
        y_pred=predictions == pos_label,
        sensitive_features=Z_test,
    )

    RESULT3 = mf.by_group

    # NOTE because of the random state it is hard to test. Needs to be improved
    # though.
    assert RESULT1 is not None
    assert RESULT2a is not None
    assert RESULT2b is not None
    assert RESULT3 is not None
