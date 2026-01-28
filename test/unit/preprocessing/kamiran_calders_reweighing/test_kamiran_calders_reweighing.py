# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import math
from contextlib import nullcontext as does_not_raise

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from fairlearn.preprocessing import KamiranCaldersReweighing

pytestmark = pytest.mark.narwhals


# ----------------------------------------------------------------------
# sklearn compatibility
# ----------------------------------------------------------------------
@parametrize_with_checks(
    [
        KamiranCaldersReweighing(drop_target=True),
        KamiranCaldersReweighing(drop_target=False),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    """Verify that the transformer passes standard scikit-learn checks."""
    check(estimator)


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
@pytest.fixture
def kamiran_example_dataframe():
    """
    Classic Kamiran-Calders toy dataset.

    Features:
        - Sex: M/F
        - Ethnicity: Native / Non-native
        - Job type: Board / Healthcare / Education
    Target:
        - Cl.: + / -
    """
    return pd.DataFrame(
        {
            "Sex": ["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"],
            "Ethnicity": [
                "Native",
                "Native",
                "Native",
                "Non-nat.",
                "Non-nat.",
                "Non-nat.",
                "Native",
                "Native",
                "Non-nat.",
                "Native",
            ],
            "Job type": [
                "Board",
                "Board",
                "Board",
                "Healthcare",
                "Healthcare",
                "Education",
                "Education",
                "Healthcare",
                "Education",
                "Board",
            ],
            "Cl.": ["+", "+", "+", "+", "-", "-", "-", "+", "-", "+"],
        }
    )


# ----------------------------------------------------------------------
# 1. Alignment with Kamiran-Calders example weights
# ----------------------------------------------------------------------
def test_weights_match_kamiran_calders_example(kamiran_example_dataframe):
    """
    Verify that reweighing produces weights matching the Kamiran-Calders
    toy example for a single sensitive feature.
    """
    X = kamiran_example_dataframe.drop(columns=["Cl."])
    y = kamiran_example_dataframe["Cl."]

    rw = KamiranCaldersReweighing(drop_target=False)
    X_tfm = rw.fit_transform(X, y, sensitive_features=["Sex"])

    # Expected weights taken from Kamiran-Calders example
    expected_weights = np.array([0.75, 0.75, 0.75, 0.75, 2.0, 0.67, 0.67, 1.5, 0.67, 1.5])

    np.testing.assert_allclose(
        X_tfm["weight"].to_numpy(),
        expected_weights,
        rtol=1e-2,
    )


# ----------------------------------------------------------------------
# 2. Probabilistic invariance test
# ----------------------------------------------------------------------
def test_reweighing_enforces_independence(kamiran_example_dataframe):
    """
    Verify that weighted joint distribution satisfies independence between
    sensitive feature and target (P(S,Y) = P(S) * P(Y)).
    """
    X = kamiran_example_dataframe.drop(columns=["Cl."])
    y = kamiran_example_dataframe["Cl."]

    rw = KamiranCaldersReweighing(drop_target=False)
    X_tfm = rw.fit_transform(X, y, sensitive_features=["Sex"]).join(y.to_frame("Cl."))

    total_weight = X_tfm["weight"].sum()

    # weighted joint distribution
    joint = X_tfm.groupby(["Sex", "Cl."])["weight"].sum().div(total_weight)

    # weighted marginals
    p_s = X_tfm.groupby("Sex")["weight"].sum().div(total_weight)
    p_y = X_tfm.groupby("Cl.")["weight"].sum().div(total_weight)

    # check independence
    for (s, y_val), p_sy in joint.items():
        assert math.isclose(
            p_sy,
            p_s[s] * p_y[y_val],
            rel_tol=1e-6,
        )


# ----------------------------------------------------------------------
# 3. Multiple sensitive features
# ----------------------------------------------------------------------
def test_multiple_sensitive_features_supported(kamiran_example_dataframe):
    """
    Verify that multiple sensitive features can be provided and that
    the weights column is added correctly.

    When multiple sensitive features are provided, Kamiran-Calders
    reweighing enforces independence between the joint sensitive
    attribute and the target, but does not guarantee marginal
    independence for individual sensitive features.
    """
    X = kamiran_example_dataframe.drop(columns=["Cl."])
    y = kamiran_example_dataframe["Cl."]

    rw = KamiranCaldersReweighing(drop_target=False)
    X_tfm = rw.fit_transform(X, y, sensitive_features=["Sex", "Ethnicity"])

    # basic invariants
    assert len(X_tfm) == len(X)
    assert "weight" in X_tfm.columns
    assert "Sex" in X_tfm.columns
    assert "Ethnicity" in X_tfm.columns


# ----------------------------------------------------------------------
# Defensive & edge-case tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ["sensitive_features", "expectation"],
    [
        (["Sex"], does_not_raise()),
        (["Ethnicity"], does_not_raise()),
        (["Sex", "Ethnicity"], does_not_raise()),
        (["Missing"], pytest.raises(ValueError, match="Sensitive feature")),
    ],
)
def test_sensitive_feature_validation(kamiran_example_dataframe, sensitive_features, expectation):
    """
    Validate that specifying invalid sensitive features raises ValueError
    and valid features pass silently.
    """
    X = kamiran_example_dataframe.drop(columns=["Cl."])
    y = kamiran_example_dataframe["Cl."]

    rw = KamiranCaldersReweighing()

    with expectation:
        rw.fit(X, y, sensitive_features=sensitive_features)


def test_transform_requires_fit(kamiran_example_dataframe):
    """
    Verify that calling transform before fit raises an exception.
    """
    X = kamiran_example_dataframe.drop(columns=["Cl."])
    rw = KamiranCaldersReweighing()

    with pytest.raises(Exception):
        rw.transform(X)


# ----------------------------------------------------------------------
# Narwhals / pandas interoperability
# ----------------------------------------------------------------------
def test_narwhals_dataframe_roundtrip(kamiran_example_dataframe):
    """
    Verify that the transformer supports Narwhals DataFrames and
    returns a Narwhals DataFrame with the weight column.
    """
    X = nw.from_native(kamiran_example_dataframe.drop(columns=["Cl."]), eager_only=True)
    y = kamiran_example_dataframe["Cl."]

    rw = KamiranCaldersReweighing()
    X_tfm = rw.fit_transform(X, y, sensitive_features=["Sex"])

    assert isinstance(X_tfm, nw.DataFrame)
    assert "weight" in X_tfm.columns
