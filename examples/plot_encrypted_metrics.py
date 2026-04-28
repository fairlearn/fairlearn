# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
==================================================
Computing Fairlearn metrics on encrypted predictions
==================================================
"""

# %%
# This notebook shows how to compute the canonical Fairlearn fairness
# metrics on **encrypted** predictions using the third-party
# `fairlearn-fhe <https://pypi.org/project/fairlearn-fhe/>`_ package
# (Apache-2.0). The motivating scenario is a regulator-facing fairness
# audit where the model owner cannot share per-row predictions with the
# auditor: ``y_pred`` crosses the trust boundary as CKKS ciphertext,
# the auditor still produces a verdict + tamper-evident audit envelope,
# and per-row predictions never leave the steward's environment.
#
# fairlearn-fhe ports the same metric names and call signatures from
# ``fairlearn.metrics``; the imports are the only difference.

# %%
# Optional dependency
# ===================
#
# This example requires ``fairlearn-fhe`` (which transitively pulls in
# ``tenseal`` for the default CKKS backend). If you have not installed
# it, the remaining cells are skipped (guarded by ``HAS_FHE``) so the
# gallery build still completes cleanly.

try:
    from fairlearn_fhe import build_context, encrypt
    from fairlearn_fhe.metrics import demographic_parity_difference as enc_dp
    from fairlearn_fhe.metrics import equal_opportunity_difference as enc_eopp
    from fairlearn_fhe.metrics import equalized_odds_difference as enc_eo

    HAS_FHE = True
except ImportError:
    print(
        "Skipping encrypted-metrics example: install fairlearn-fhe to run it.\n"
        "    pip install fairlearn-fhe"
    )
    HAS_FHE = False

# %%
# Train a baseline classifier
# ===========================
#
# We use the same UCI Adult dataset as the existing
# ``plot_make_derived_metric`` example. The trained model is a plain
# logistic regression — fairlearn-fhe does not constrain the model
# choice; only the post-hoc metric evaluation runs under encryption.

if HAS_FHE:
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from fairlearn.datasets import fetch_adult
    from fairlearn.metrics import (
        demographic_parity_difference,
        equal_opportunity_difference,
        equalized_odds_difference,
    )

    data = fetch_adult(as_frame=True)
    X_raw = data.data
    y_true = (data.target == ">50K").astype(int)
    sensitive = X_raw["sex"].astype(str)

    X = X_raw.drop(columns=["sex"])
    X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
        X, y_true, sensitive, test_size=0.3, random_state=12345, stratify=y_true
    )

    numeric = X_train.select_dtypes(include="number").columns
    categorical = X_train.select_dtypes(exclude="number").columns
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))]).fit(
        X_train, y_train
    )
    y_pred = model.predict(X_test)
    print(f"Trained on {len(X_train)} rows, evaluating on {len(y_pred)}.")

# %%
# Plaintext baseline
# ==================
#
# Compute the canonical fairness metrics in plaintext so the encrypted
# verdicts have a reference value.

if HAS_FHE:
    plain = {
        "demographic_parity_difference": demographic_parity_difference(
            y_test, y_pred, sensitive_features=sf_test
        ),
        "equalized_odds_difference": equalized_odds_difference(
            y_test, y_pred, sensitive_features=sf_test
        ),
        "equal_opportunity_difference": equal_opportunity_difference(
            y_test, y_pred, sensitive_features=sf_test
        ),
    }
    print(pd.Series(plain, name="plaintext").to_string())

# %%
# Encrypted run
# =============
#
# Build a CKKS context, encrypt ``y_pred`` only, and call the same-named
# helpers from ``fairlearn_fhe.metrics``. ``y_test`` and ``sf_test``
# remain plaintext on the auditor side — this is *Mode A* in
# fairlearn-fhe's trust-models guide (depth-1 CKKS circuit per metric).

if HAS_FHE:
    ctx = build_context(backend="tenseal")
    y_pred_enc = encrypt(ctx, y_pred.astype(float))

    encrypted = {
        "demographic_parity_difference": enc_dp(y_test, y_pred_enc, sensitive_features=sf_test),
        "equalized_odds_difference": enc_eo(y_test, y_pred_enc, sensitive_features=sf_test),
        "equal_opportunity_difference": enc_eopp(y_test, y_pred_enc, sensitive_features=sf_test),
    }
    print(pd.Series(encrypted, name="encrypted").to_string())

# %%
# Numerical agreement
# ===================
#
# CKKS is an approximate scheme; small rounding noise is expected. The
# default settings give ``< 1e-3`` absolute error in practice.

if HAS_FHE:
    import math

    for name in plain:
        delta = abs(plain[name] - encrypted[name])
        flag = "OK" if delta < 1e-3 else "!!"
        print(
            f"{flag} {name}: plain={plain[name]:.6f} fhe={encrypted[name]:.6f} " f"|Δ|={delta:.2e}"
        )
        assert math.isclose(plain[name], encrypted[name], abs_tol=1e-3)

# %%
# Audit envelope
# ==============
#
# fairlearn-fhe also produces a JSON-serialisable ``MetricEnvelope``
# capturing the parameter-set hash, observed depth, op counts, input
# hashes, and a UTC timestamp. The envelope can be Ed25519-signed and
# validated from the CLI without importing an FHE backend
# (``fairlearn-fhe verify envelope.json``).

if HAS_FHE:
    from fairlearn_fhe import audit_metric

    env = audit_metric(
        "demographic_parity_difference",
        y_true=y_test,
        y_pred=y_pred_enc,
        sensitive_features=sf_test,
        ctx=ctx,
        min_group_size=30,
    )
    env_dict = env.to_dict()
    print(
        {
            k: env_dict[k]
            for k in [
                "metric_name",
                "value",
                "observed_depth",
                "op_counts",
                "n_samples",
                "n_groups",
            ]
        }
    )

# %%
# Trust models
# ============
#
# - **Mode A** (above): ``y_pred`` encrypted, ``y_true`` and
#   ``sensitive_features`` plaintext. Depth-1 circuit per metric.
# - **Mode B**: ``y_pred`` and the per-row group-membership masks are
#   both encrypted. Depth-2 circuit. Auditor sees only group counts as
#   plaintext metadata. See ``fairlearn_fhe.encrypt_sensitive_features``.
#
# fairlearn-fhe's ``threat-model.md`` formalises what the auditor learns
# vs does not learn under each mode. Mode A is the recommended default
# for routine fairness audits; Mode B is for regulator-facing audits
# where the auditor cannot be trusted to handle per-row sensitive
# features.
