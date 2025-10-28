# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import sys
from typing import Literal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from fairlearn.adversarial import (
    AdversarialFairnessClassifier,
    AdversarialFairnessRegressor,
)
from fairlearn.adversarial._adversarial_mitigation import _AdversarialFairness
from fairlearn.adversarial._preprocessor import FloatTransformer
from fairlearn.utils._fixes import parametrize_with_checks

from .helper import (
    Bin1d,
    Bin2d,
    Cat,
    Cont2d,
    KeywordToClass,
    MultiClass2d,
    cols,
    generate_data_combinations,
    get_backend_patches,
    get_instance,
    get_instance_with_context,
)

BackendType = Literal["torch", "tensorflow"]


@pytest.fixture(scope="function")
def fake_backend_env(request: pytest.FixtureRequest):
    """Patch sys.modules depending on the backend."""

    backend = request.param

    with patch.dict(sys.modules, get_backend_patches(backend)):
        yield backend


@pytest.mark.parametrize("fake_backend_env", ["torch", "tensorflow"], indirect=True)
def test_model_init(fake_backend_env):
    """Test model init from list."""

    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        fake_backend=fake_backend_env,
        fake_mixin=False,
        fake_training=True,
        predictor_model=[10, "Sigmoid", "Softmax", "ReLU", "Leaky_ReLU"],
    )
    mitigator.fit(X, Y, sensitive_features=Z)
    layers = mitigator.backendEngine_.predictor_model.layers_
    if hasattr(layers, "layers"):
        layers = layers.layers
    assert not hasattr(layers[0], "a") or layers[0].a == cols
    assert layers[0].b == 10
    assert layers[1].__name__ == "Sigmoid"
    assert layers[2].__name__ == "Softmax"
    assert layers[3].__name__ == "ReLU"
    assert layers[4].__name__.replace("_", "") == "LeakyReLU"
    assert not hasattr(layers[0], "a") or layers[5].a == 10
    assert layers[5].b == 1
    assert layers[6].__name__.lower() == "sigmoid"
    assert len(layers) == 7


@pytest.mark.parametrize("fake_backend_env", ["torch", "tensorflow"], indirect=True)
def test_model_params(fake_backend_env):
    """Test training params."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        fake_backend=fake_backend_env,
        fake_mixin=False,
        fake_training=True,
        predictor_model=[10, "ReLU"],
        adversary_model=[3, "ReLU"],
        warm_start=True,
        shuffle=True,
        learning_rate=0.1,
        epochs=3,
        batch_size=3,
        max_iter=10,
        random_state=1,
        progress_updates=0.0000001,
        callbacks=[lambda *args, **kwargs: False, lambda *args, **kwargs: False],
    )
    mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("fake_backend_env", ["torch", "tensorflow"], indirect=True)
def test_model_early_stop(fake_backend_env):
    """Test training params."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        fake_backend=fake_backend_env,
        fake_mixin=False,
        fake_training=True,
        predictor_model=[10, "ReLU"],
        adversary_model=[3, "ReLU"],
        warm_start=True,
        shuffle=True,
        learning_rate=0.1,
        epochs=3,
        batch_size=3,
        max_iter=10,
        random_state=1,
        progress_updates=0.0000001,
        callbacks=lambda callback_obj, step, *args, **kwargs: step > 5,
    )
    mitigator.fit(X, Y, sensitive_features=Z)
    assert mitigator.n_iter_ == 6


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_model_equalized_odds_model_setup(fake_backend_env):
    """Test model initialization with equalized_odds."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        fake_backend=fake_backend_env,
        fake_mixin=False,
        fake_training=True,
        predictor_model=[10, "ReLU"],
        adversary_model=[3, "ReLU"],
        constraints="equalized_odds",
    )
    mitigator.fit(X, Y, sensitive_features=Z)
    assert mitigator.backendEngine_.adversary_model.layers_.layers[0].a == 2


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_model_kw_error_constraint(fake_backend_env):
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            fake_backend=fake_backend_env,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            constraints="hihi",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_model_kw_error_ints(fake_backend_env):
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            fake_backend=fake_backend_env,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            batch_size=-0.5,
            max_iter=-0.1,
        )
        mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_model_kw_error_bools(fake_backend_env):
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            fake_backend=fake_backend_env,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            shuffle="true",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_model_kw_error_callback(fake_backend_env):
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            fake_backend=fake_backend_env,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            callbacks="hi",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_model_kw_error_callbacks(fake_backend_env):
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            fake_backend=fake_backend_env,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            callbacks=[lambda x: x, "bye"],
        )
        mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_model_kw_error_tf(fake_backend_env):
    """Test kw error."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    other = get_instance(
        fake_backend="tensorflow",
        fake_mixin=False,
        fake_training=True,
    )

    with pytest.raises(ValueError):
        mitigator = get_instance(
            fake_backend="torch",
            fake_mixin=False,
            fake_training=True,
            predictor_model=other.predictor_model,
            adversary_model=other.predictor_model,
            backend="tensorflow",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("fake_backend_env", ["tensorflow"], indirect=True)
def test_model_kw_error_torch(fake_backend_env):
    """Test kw error."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    other = get_instance(
        fake_backend="torch",
        fake_mixin=False,
        fake_training=True,
    )

    with pytest.raises(ValueError):
        mitigator = get_instance(
            fake_backend="tensorflow",
            fake_mixin=False,
            fake_training=True,
            predictor_model=other.predictor_model,
            adversary_model=other.adversary_model,
            backend="torch",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


EXPECTED_FAILED_CHECKS = {
    "AdversarialFairnessClassifier": {
        "check_estimators_pickle": "pickling is not possible.",
        "check_estimators_overwrite_params": "pickling is not possible.",
    },
    "AdversarialFairnessRegressor": {
        "check_estimators_pickle": "pickling is not possible.",
        "check_estimators_overwrite_params": "pickling is not possible.",
    },
}


@pytest.mark.parametrize("fake_backend_env", ["tensorflow"], indirect=True)
@parametrize_with_checks(
    [
        get_instance_with_context(
            AdversarialFairnessClassifier,
            fake_training=True,
            fake_backend="tensorflow",
            fake_mixin=True,
        ),
        get_instance_with_context(
            AdversarialFairnessRegressor,
            fake_training=True,
            fake_backend="tensorflow",
            fake_mixin=True,
        ),
    ],
    expected_failed_checks=lambda x: EXPECTED_FAILED_CHECKS.get(x.__class__.__name__, {}),
)
def test_estimators_with_tensorflow(estimator, check, fake_backend_env):
    """Check the compatibility with scikit-learn API."""
    check(estimator)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
@parametrize_with_checks(
    [
        get_instance(
            _AdversarialFairness,
            fake_backend="torch",
            fake_mixin=True,
        ),
        get_instance(
            AdversarialFairnessClassifier,
            fake_training=True,
            fake_backend="torch",
            fake_mixin=True,
        ),
        get_instance(
            AdversarialFairnessRegressor,
            fake_training=True,
            fake_backend="torch",
            fake_mixin=True,
        ),
    ],
    expected_failed_checks=lambda x: EXPECTED_FAILED_CHECKS.get(x.__class__.__name__, {}),
)
def test_estimators_with_torch(estimator, check, fake_backend_env):
    """Check the compatibility with scikit-learn API."""
    check(estimator)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_fake_models(fake_backend_env):
    """Test various data types and see if it is interpreted correctly."""
    for (X, Y, Z), (X_type, Y_type, Z_type) in generate_data_combinations():
        mitigator = get_instance(fake_training=True, fake_backend=fake_backend_env)
        mitigator.fit(X, Y, sensitive_features=Z)
        assert isinstance(mitigator.backendEngine_.predictor_loss, KeywordToClass(Y_type))
        assert isinstance(mitigator.backendEngine_.adversary_loss, KeywordToClass(Z_type))


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_fake_models_list_inputs(fake_backend_env):
    """Test model with lists as input."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(X.tolist(), Y.tolist(), sensitive_features=Z.tolist())


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_fake_models_df_inputs(fake_backend_env):
    """Test model with data frames as input."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(pd.DataFrame(X), pd.Series(Y), sensitive_features=pd.DataFrame(Z))


@pytest.mark.parametrize(
    "data, valid_choice",
    [
        (Bin2d, "binary"),
        (Cat, "binary"),
        (Cont2d, "continuous"),
        (MultiClass2d, "multiclass"),
    ],
)
def test_valid_input_data_types(data, valid_choice):
    """Test if the model processes the right data types"""
    prep = FloatTransformer(transformer=valid_choice)
    prep.fit(data)
    assert prep.n_features_in_ == data.shape[0]
    assert prep.n_features_out_ == data.shape[1]


def check_2dnp(X):
    """Make sure X is a 2d float ndarray."""
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert len(X.shape) == 2
    assert X.dtype == float


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_validate_data(fake_backend_env):
    """Test if validate_data properly preprocesses datasets to ndarray."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X, Y, Z)
        for x in (X, Y, Z):
            check_2dnp(x)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_validate_data_list_inputs(fake_backend_env):
    """Test if validate_data properly preprocesses list datasets to ndarray."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X.tolist(), Y.tolist(), Z.tolist())
        for x in (X, Y, Z):
            check_2dnp(x)


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_validate_data_df_inputs(fake_backend_env):
    """Test if validate_data properly preprocesses dataframes to ndarray."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(pd.DataFrame(X), pd.Series(Y), pd.DataFrame(Z))
        for x in (X, Y, Z):
            check_2dnp(x)


@pytest.mark.parametrize("fake_backend_env", ["torch", "tensorflow"], indirect=True)
def test_not_correct_backend(fake_backend_env):
    """Test if validate_data properly preprocesses dataframes to ndarray."""
    if fake_backend_env == "torch":
        backend = "tensorflow"
    else:
        backend = "torch"

    mitigator = get_instance(
        fake_backend=fake_backend_env,
        backend=backend,
    )
    with pytest.raises(RuntimeError):
        mitigator._validate_backend()


@pytest.mark.parametrize("fake_backend_env", [None], indirect=True)
@pytest.mark.parametrize("backend", ["torch", "tensorflow"])
def test_no_backend(backend, fake_backend_env):
    """Test if validate_data properly preprocesses dataframes to ndarray."""
    mitigator = get_instance(fake_backend=fake_backend_env, backend=backend)
    with pytest.raises(RuntimeError):
        mitigator._validate_backend()


@pytest.mark.parametrize("fake_backend_env", ["torch"], indirect=True)
def test_validate_data_ambiguous_rows(fake_backend_env):
    """Test if an ambiguous number of rows are caught."""
    for (X, Y, Z), types in generate_data_combinations():
        X = X[:5, :]
        mitigator = get_instance(fake_mixin=True)
        with pytest.raises(ValueError) as exc:
            mitigator._validate_input(X.tolist(), Y.tolist(), Z.tolist())
            assert str(
                exc.value
            ) == "Input data has an ambiguous number of rows: {}, {}, {}.".format(
                X.shape[0], Y.shape[0], Z.shape[0]
            )
