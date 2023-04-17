# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import sys
import pytest
import numpy as np
import pandas as pd
from sklearn.utils import estimator_checks
from fairlearn.adversarial import (
    AdversarialFairnessClassifier,
    AdversarialFairnessRegressor,
)
from fairlearn.adversarial._adversarial_mitigation import _AdversarialFairness
from fairlearn.adversarial._preprocessor import FloatTransformer
from fairlearn.adversarial._constants import _TYPE_COMPLIANCE_ERROR

from .helper import (
    get_instance,
    generate_data_combinations,
    cols,
    Bin2d,
    Bin1d,
    Cat,
    Cont2d,
    Cont1d,
    BCE,
    CCE,
    MSE,
    Keyword_CATEGORY,
    Keyword_BINARY,
    Keyword_CONTINUOUS,
    Keyword_CLASSIFICATION,
    Keyword_AUTO,
    KeywordToClass,
)


@pytest.mark.parametrize("torch", [True, False])
def test_model_init(torch):
    """Test model init from list."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        torch=torch,
        tensorflow=not torch,
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


@pytest.mark.parametrize("torch", [True, False])
def test_model_params(torch):
    """Test training params."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        torch=torch,
        tensorflow=not torch,
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
        callbacks=[lambda *args: False, lambda *args: False],
    )
    mitigator.fit(X, Y, sensitive_features=Z)


@pytest.mark.parametrize("torch", [True, False])
def test_model_early_stop(torch):
    """Test training params."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        torch=torch,
        tensorflow=not torch,
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
        callbacks=lambda object, step: step > 5,
    )
    mitigator.fit(X, Y, sensitive_features=Z)
    assert mitigator.step_ == 6


def test_model_equalized_odds_model_setup():
    """Test model initialization with equalized_odds."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    mitigator = get_instance(
        torch=True,
        tensorflow=False,
        fake_mixin=False,
        fake_training=True,
        predictor_model=[10, "ReLU"],
        adversary_model=[3, "ReLU"],
        constraints="equalized_odds",
    )
    mitigator.fit(X, Y, sensitive_features=Z)
    assert mitigator.backendEngine_.adversary_model.layers_.layers[0].a == 2


def test_model_kw_error_constraint():
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            torch=True,
            tensorflow=False,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            constraints="hihi",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


def test_model_kw_error_ints():
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            torch=True,
            tensorflow=False,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            batch_size=-0.5,
            max_iter=-0.1,
        )
        mitigator.fit(X, Y, sensitive_features=Z)


def test_model_kw_error_bools():
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            torch=True,
            tensorflow=False,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            shuffle="true",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


def test_model_kw_error_callback():
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            torch=True,
            tensorflow=False,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            callbacks="hi",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


def test_model_kw_error_callbacks():
    """Test kw error."""
    with pytest.raises(ValueError):
        (X, Y, Z) = Bin2d, Bin1d, Bin1d
        mitigator = get_instance(
            torch=True,
            tensorflow=False,
            fake_mixin=False,
            fake_training=True,
            predictor_model=[10, "ReLU"],
            adversary_model=[3, "ReLU"],
            callbacks=[lambda x: x, "bye"],
        )
        mitigator.fit(X, Y, sensitive_features=Z)


def test_model_kw_error_tf():
    """Test kw error."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    other = get_instance(
        torch=False,
        tensorflow=True,
        fake_mixin=False,
        fake_training=True,
    )

    with pytest.raises(ValueError):
        mitigator = get_instance(
            torch=True,
            tensorflow=False,
            fake_mixin=False,
            fake_training=True,
            predictor_model=other.predictor_model,
            adversary_model=other.predictor_model,
            backend="tensorflow",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


def test_model_kw_error_torch():
    """Test kw error."""
    (X, Y, Z) = Bin2d, Bin1d, Bin1d
    other = get_instance(
        torch=True,
        tensorflow=False,
        fake_mixin=False,
        fake_training=True,
    )

    with pytest.raises(ValueError):
        mitigator = get_instance(
            torch=False,
            tensorflow=True,
            fake_mixin=False,
            fake_training=True,
            predictor_model=other.predictor_model,
            adversary_model=other.adversary_model,
            backend="torch",
        )
        mitigator.fit(X, Y, sensitive_features=Z)


# The following list of checks was generated by calling
# estimator_checks.check_estimator on our estimator with
# parameter generate_only=True
@pytest.mark.parametrize(
    "test_fn",
    [
        estimator_checks.check_no_attributes_set_in_init,
        estimator_checks.check_estimators_dtypes,
        estimator_checks.check_fit_score_takes_y,
        estimator_checks.check_estimators_fit_returns_self,
        # The following does check seems equivalent to the previous, and it
        # requires a parameter: readonly_memmap=True. So, we skip it.
        # estimator_checks.check_estimators_fit_returns_self
        estimator_checks.check_complex_data,
        estimator_checks.check_dtype_object,
        estimator_checks.check_estimators_empty_data_messages,
        estimator_checks.check_pipeline_consistency,
        estimator_checks.check_estimators_nan_inf,
        # The following two checks do not work, because pickling does not work.
        # estimator_checks.check_estimators_overwrite_params,
        estimator_checks.check_estimator_sparse_data,
        # estimator_checks.check_estimators_pickle,
        estimator_checks.check_estimator_get_tags_default_keys,
        # The following check seems impossible, because we cannot have an
        # empty list as default parameter.
        # Possible fix: set default param to None, and in fit change None to [].
        # estimator_checks.check_parameters_default_constructible,
        estimator_checks.check_methods_sample_order_invariance,
        estimator_checks.check_methods_subset_invariance,
        estimator_checks.check_fit2d_1sample,
        estimator_checks.check_fit2d_1feature,
        estimator_checks.check_get_params_invariance,
        estimator_checks.check_set_params,
        estimator_checks.check_dict_unchanged,
        estimator_checks.check_dont_overwrite_parameters,
        estimator_checks.check_fit_idempotent,
        estimator_checks.check_fit_check_is_fitted,
        estimator_checks.check_n_features_in,
        estimator_checks.check_fit1d,
        estimator_checks.check_fit2d_predict1d,
    ],
)
def test_estimator_checks(test_fn):
    instance = get_instance(
        _AdversarialFairness, torch=True, tensorflow=False, fake_mixin=True
    )

    test_fn(
        _AdversarialFairness.__name__,
        instance,
    )


@pytest.mark.parametrize("torch1", [True, False])
def test_classifier(torch1):
    """Test classifier subclass."""
    mitigator = get_instance(
        AdversarialFairnessClassifier,
        fake_training=True,
        torch=torch1,
        tensorflow=not torch1,
    )
    assert isinstance(mitigator, _AdversarialFairness)

    mitigator.fit(Cont2d, Bin1d, sensitive_features=Cat)
    assert isinstance(mitigator.backendEngine_.predictor_loss, BCE)
    assert isinstance(mitigator.backendEngine_.adversary_loss, CCE)


@pytest.mark.parametrize("torch2", [True, False])
def test_regressor(torch2):
    """Test regressor subclass."""
    mitigator = get_instance(
        AdversarialFairnessRegressor,
        fake_training=True,
        torch=torch2,
        tensorflow=not torch2,
    )
    assert isinstance(mitigator, _AdversarialFairness)

    mitigator.fit(Cont2d, Cont2d, sensitive_features=Cat)
    assert isinstance(mitigator.backendEngine_.predictor_loss, MSE)
    assert isinstance(mitigator.backendEngine_.adversary_loss, CCE)


@pytest.mark.parametrize("torch3", [True, False])
def test_fake_models(torch3):
    """Test various data types and see if it is interpreted correctly."""
    for (X, Y, Z), (X_type, Y_type, Z_type) in generate_data_combinations():
        mitigator = get_instance(
            fake_training=True, torch=torch3, tensorflow=not torch3
        )

        mitigator.fit(X, Y, sensitive_features=Z)
        assert isinstance(
            mitigator.backendEngine_.predictor_loss, KeywordToClass(Y_type)
        )
        assert isinstance(
            mitigator.backendEngine_.adversary_loss, KeywordToClass(Z_type)
        )


def test_fake_models_list_inputs():
    """Test model with lists as input."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(X.tolist(), Y.tolist(), sensitive_features=Z.tolist())


def test_fake_models_df_inputs():
    """Test model with data frames as input."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(
            pd.DataFrame(X), pd.DataFrame(Y), sensitive_features=pd.DataFrame(Z)
        )


def check_type_helper(data, actual_type, valid_choices, invalid_choices):
    """Help to check if distribution types are interpreted correctly."""
    for valid_choice in valid_choices:
        prep = FloatTransformer(transformer=valid_choice)
        prep.fit(data)
        assert prep.dist_type_ == actual_type

    for invalid_choice in invalid_choices:
        with pytest.raises(ValueError) as exc:
            prep = FloatTransformer(transformer=invalid_choice)
            prep.fit(data)
            assert str(exc.value) == _TYPE_COMPLIANCE_ERROR.format(
                invalid_choice, prep.inferred_type_
            )


def test_check_type_correct_data():
    """Test distribution types on some correct/incorrectly distributed data."""
    check_type_helper(
        Bin1d,
        Keyword_BINARY,
        [Keyword_AUTO, Keyword_BINARY, Keyword_CLASSIFICATION],
        [Keyword_CATEGORY, None, "bogus"],
    )
    check_type_helper(
        Cat,
        Keyword_CATEGORY,
        [Keyword_AUTO, Keyword_CATEGORY, Keyword_CLASSIFICATION],
        [Keyword_BINARY, None, "bogus"],
    )
    check_type_helper(
        Cont1d,
        Keyword_CONTINUOUS,
        [Keyword_AUTO, Keyword_CONTINUOUS, None],
        [
            Keyword_BINARY,
            Keyword_CATEGORY,
            Keyword_CLASSIFICATION,
            "bogus",
        ],
    )
    check_type_helper(
        Cont2d,
        Keyword_CONTINUOUS,
        [Keyword_AUTO, Keyword_CONTINUOUS, None],
        [
            Keyword_BINARY,
            Keyword_CATEGORY,
            Keyword_CLASSIFICATION,
            "bogus",
        ],
    )
    # Bin2d is not interpretable as binary 2d, nor continuous, because it is so ambiguous.
    check_type_helper(
        Bin2d,
        None,
        [],
        [
            Keyword_BINARY,
            Keyword_CATEGORY,
            Keyword_CLASSIFICATION,
            Keyword_CONTINUOUS,
            None,
            "bogus",
        ],
    )

    with pytest.raises(ValueError) as exc:
        prep = FloatTransformer(transformer=Keyword_AUTO)
        prep.fit(Bin2d)
        assert str(exc.value) == _TYPE_COMPLIANCE_ERROR.format(
            Keyword_AUTO, prep.inferred_type_
        )


def test_check_type_faulty_data():
    """Check distribution types on slightly faulty datasets."""
    notBin1d = Bin1d.copy()
    notBin1d[0] = 0.1
    check_type_helper(
        notBin1d,
        Keyword_CONTINUOUS,
        [Keyword_AUTO, Keyword_CONTINUOUS],
        [Keyword_BINARY, Keyword_CATEGORY, Keyword_CLASSIFICATION],
    )
    notCat = Cat.copy()
    notCat[0, 0:2] = 1
    # Special case where values are still {0,1}, but sums arent 1. Then we also
    # reject continuous
    check_type_helper(
        notCat,
        None,
        [],  # Auto doesnt work on ambiguous
        [
            Keyword_BINARY,
            Keyword_CATEGORY,
            Keyword_CLASSIFICATION,
            Keyword_CONTINUOUS,
            Keyword_AUTO,
        ],
    )
    notCat[0, 0] = 0.5
    check_type_helper(
        notCat,
        Keyword_CONTINUOUS,
        [Keyword_AUTO, Keyword_CONTINUOUS],  # Auto does work here
        [Keyword_BINARY, Keyword_CATEGORY, Keyword_CLASSIFICATION],
    )
    notCat[0, 1] = 0.5
    notCat[0, 2:] = (
        0.0  # Special case because now first row sums to one but is not one-hot
    )
    check_type_helper(
        notCat,
        Keyword_CONTINUOUS,
        [Keyword_AUTO, Keyword_CONTINUOUS],  # Auto does work here
        [Keyword_BINARY, Keyword_CATEGORY, Keyword_CLASSIFICATION],
    )


def check_2dnp(X):
    """Make sure X is a 2d float ndarray."""
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert len(X.shape) == 2
    assert X.dtype == float


def test_validate_data():
    """Test if validate_data properly preprocesses datasets to ndarray."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X, Y, Z)
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_list_inputs():
    """Test if validate_data properly preprocesses list datasets to ndarray."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X.tolist(), Y.tolist(), Z.tolist())
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_df_inputs():
    """Test if validate_data properly preprocesses dataframes to ndarray."""
    for (X, Y, Z), types in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(
            pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(Z)
        )
        for x in (X, Y, Z):
            check_2dnp(x)


@pytest.mark.parametrize("backend", ["torch", "tensorflow"])
def test_not_correct_backend(backend):
    """Test if validate_data properly preprocesses dataframes to ndarray."""
    mitigator = get_instance(
        torch=backend != "torch",
        tensorflow=backend != "tensorflow",
        backend=backend,
    )
    with pytest.raises(RuntimeError):
        mitigator._validate_backend()


@pytest.mark.parametrize("backend", ["torch", "tensorflow"])
def test_no_backend(backend):
    """Test if validate_data properly preprocesses dataframes to ndarray."""
    mitigator = get_instance(torch=False, tensorflow=False, backend=backend)
    with pytest.raises(RuntimeError):
        mitigator._validate_backend()


def test_validate_data_ambiguous_rows():
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


sys.modules["torch"] = None
sys.modules["tensorflow"] = None
