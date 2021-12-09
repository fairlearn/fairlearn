# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import sys  # , fake_torch
import pytest
import numpy as np
import pandas as pd
from sklearn.utils import estimator_checks
from fairlearn.adversarial import (
    AdversarialFairness,
    AdversarialFairnessClassifier,
    AdversarialFairnessRegressor,
)
from fairlearn.adversarial._preprocessor import FloatTransformer
from fairlearn.adversarial._pytorch_engine import PytorchEngine
from fairlearn.adversarial._tensorflow_engine import TensorflowEngine
from fairlearn.adversarial._backend_engine import BackendEngine
from fairlearn.adversarial._constants import _TYPE_COMPLIANCE_ERROR


model_class = type("Model", (object,), {})


Keyword_BINARY = "binary"
Keyword_CATEGORY = "category"
Keyword_CONTINUOUS = "continuous"
Keyword_AUTO = "auto"
Keyword_CLASSIFICATION = "classification"


class loss_class:
    """Mock loss class."""

    def __init__(self, **kwargs):
        """Do not need to init anything."""
        pass

    # class loss:
    #     def backward(self, **kwargs): return
    def __call__(self, x, y):
        """Mock loss should be callable from train_step."""
        return  # self.loss()


BCE = type(Keyword_BINARY, (loss_class,), {})
CCE = type(Keyword_CATEGORY, (loss_class,), {})
MSE = type(Keyword_CONTINUOUS, (loss_class,), {})


def KeywordToClass(kw):
    """Map keyword (string) to a loss class."""
    index = [Keyword_BINARY, Keyword_CATEGORY, Keyword_CONTINUOUS].index(kw)
    return [BCE, CCE, MSE][index]


layer_class = type("Layer", (object,), {})


class fake_torch:
    """Mock of the PyTorch module."""

    class nn:
        """Mock of torch.nn."""

        Module = type(
            "Module",
            (model_class,),
            {
                "forward": lambda self, x: x,
                "parameters": lambda self: (i for i in [1, 2]),
                "train": lambda self: None,
                "__call__": lambda self, X: self.forward(X),
            },
        )
        BCELoss = type("BCELoss", (BCE,), {})
        NLLLoss = type("NLLLoss", (CCE,), {})
        MSELoss = type("MSELoss", (MSE,), {})

    class optim:
        """Mock of torch.optim."""

        class Optimizer:
            """Mock of pytorch Adam optimizer, with empty init and zerograd."""

            def __init__(self, params, **kwargs):
                """Mock init because this mock optimizer is initialized."""
                pass

            def zero_grad(self):
                """Mock zerograd because this mock optimizer is used."""
                pass
        
        class Adam(Optimizer): pass

    class from_numpy:
        """Mock torch.from_numpy, torch.Tensor is now a class with data."""

        def __init__(self, X):
            """Save data."""
            self.X = X

        def float(self):  # noqa: A003
            """Because it is used in the algorithm."""
            return self.X

    def manual_seed(seed):
        """Mock of set seed."""
        pass


class fake_tensorflow:
    """Mock of the TensorFlow module."""

    class keras:
        """Mock of tf.keras."""

        Model = type("Model", (model_class,), {})

        class losses:
            """Mock of tf.keras.losses."""

            BinaryCrossentropy = type("BinaryCrossentropy", (BCE,), {})
            CategoricalCrossentropy = type(
                "CategoricalCrossentropy", (CCE,), {}
            )
            MeanSquaredError = type("MeanSquaredError", (MSE,), {})

        class optimizers:
            """Mock of tf.keras.optimizers."""
            class Optimizer:
                """Mock of pytorch Adam optimizer, with empty init."""

                def __init__(self, **kwargs):  # noqa: D107
                    pass

            class Adam(Optimizer): pass

    class random:
        """mock of tf.random."""

        def set_seed(seed):
            """Set the random seed."""
            pass


rows = 60
cols = 5
Bin2d = np.random.choice([0.0, 1.0], size=(rows, cols))
Bin1d = np.random.choice([0.0, 1.0], size=(rows, 1))
Cat = np.zeros((rows, cols), dtype=float)
Cat[
    np.arange(rows), np.random.choice([i for i in range(cols)], size=(rows,))
] = 1.0
Cont2d = np.random.rand(rows, cols)
Cont1d = np.random.rand(rows, 1)


def generate_data_combinations(n=10):
    """
    Generate datasets with appropriate (and random) distribution type.

    Yields
    ------
    ((X, Y, Z), (X_type, Y_type, Z_type)) : tuple
        (X, Y, Z) is data, and (X_type, Y_type, Z_type) are their respective
        distribution types.
    """
    datas = [Bin1d, Cat, Cont2d, Cont1d]
    dist_type = [
        Keyword_BINARY,
        Keyword_CATEGORY,
        Keyword_CONTINUOUS,
        Keyword_CONTINUOUS,
    ]
    K = len(datas)
    total_combinations = K ** 3
    combinations = np.random.choice(total_combinations, size=n).tolist()
    for c in combinations:
        c_orig = c
        X = c % K
        c = (c - X) // K
        Y = c % K
        c = (c - Y) // K
        Z = c % K
        assert X + Y * K + Z * K * K == c_orig
        X_type = dist_type[X]
        X = datas[X]
        Y_type = dist_type[Y]
        Y = datas[Y]
        Z_type = dist_type[Z]
        Z = datas[Z]
        yield (X, Y, Z), (X_type, Y_type, Z_type)


class RemoveAll(BackendEngine):
    """Mock BackendEngine that implements nothing, to test base class."""

    def __init__(self, base, X, Y, Z):  # noqa: D107
        self.base = base
        pass

    def evaluate(self, X):
        """Deterministic evaluation function."""
        cols = self.base.y_transform_.n_features_out_
        rows = len(X)
        y = []
        for row in range(rows):
            rng = np.random.default_rng(
                int(np.round(np.mean(X[row]) * (2 ** 32)))
            )
            y.append(rng.random(cols))
        return np.stack(y)

    def train_step(self, X, Y, Z):  # noqa: D102
        return (0, 0)

    def get_optimizer(self, optimizer, model):  # noqa: D102
        pass

    def get_loss(self, Y, choice, data_name):  # noqa: D102
        pass


class RemoveTrainStepPytorch(PytorchEngine):
    """Mock train_step only, then we can still perform Backend inits."""

    def train_step(self, X, Y, Z):  # noqa: D102
        return (0, 0)


class RemoveTrainStepTensorflow(TensorflowEngine):
    """Mock train_step only, then we can still perform Backend inits."""

    def train_step(self, X, Y, Z):  # noqa: D102
        return (0, 0)


def get_instance(
    cls=AdversarialFairness,
    fake_mixin=False,
    fake_training=False,
    torch=True,
    tensorflow=False,
):
    """
    Shared set up of test cases that create an instance of the model.

    Parameters
    ----------
    cls : class
        class to initialize, should be (sub)class (of) AdversarialFairness

    fake_mixin: bool
        use an entirely fake backendEngine

    fake_training: bool
        only remove training step from backendEngine

    torch: bool
        Use torch (and set the fake torch module)

    tensorflow: bool
        Use tensorflow (and set the fake tensorflow module)
    """
    if torch:
        sys.modules["torch"] = fake_torch
    else:
        sys.modules["torch"] = None
    if tensorflow:
        sys.modules["tensorflow"] = fake_tensorflow
    else:
        sys.modules["tensorflow"] = None

    if torch:
        predictor = fake_torch.nn.Module()
        adversary = fake_torch.nn.Module()
    elif tensorflow:
        predictor = fake_tensorflow.keras.Model()
        adversary = fake_tensorflow.keras.Model()

    mitigator = cls(predictor_model=predictor, adversary_model=adversary)

    if fake_mixin:
        mitigator.backend = RemoveAll
    elif fake_training:
        if tensorflow:
            mitigator.backend = RemoveTrainStepTensorflow
        if torch:
            mitigator.backend = RemoveTrainStepPytorch

    return mitigator


# TODO: extend list or work on commented-out checks.
@pytest.mark.parametrize(
    "test_fn",
    [
        # transformer checks
        # estimator_checks.check_transformer_general,
        # estimator_checks.check_transformers_unfitted,
        # general estimator checks
        estimator_checks.check_fit2d_predict1d,
        estimator_checks.check_methods_subset_invariance,
        estimator_checks.check_fit2d_1sample,
        estimator_checks.check_fit2d_1feature,
        estimator_checks.check_get_params_invariance,
        estimator_checks.check_set_params,
        estimator_checks.check_dict_unchanged,
        estimator_checks.check_dont_overwrite_parameters,
        # nonmeta_checks
        estimator_checks.check_estimators_dtypes,
        estimator_checks.check_fit_score_takes_y,
        estimator_checks.check_dtype_object,
        # estimator_checks.check_sample_weights_pandas_series,
        # estimator_checks.check_sample_weights_list,
        estimator_checks.check_estimators_fit_returns_self,
        estimator_checks.check_complex_data,
        estimator_checks.check_estimators_empty_data_messages,
        estimator_checks.check_pipeline_consistency,
        estimator_checks.check_estimators_nan_inf,
        # estimator_checks.check_estimators_overwrite_params,
        estimator_checks.check_estimator_sparse_data,
        # estimator_checks.check_estimators_pickle,
    ],
)
def test_estimator_checks(test_fn):
    instance = get_instance(
        AdversarialFairness, torch=True, tensorflow=False, fake_mixin=True
    )
    test_fn(
        AdversarialFairness.__name__,
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
    assert isinstance(mitigator, AdversarialFairness)

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
    assert isinstance(mitigator, AdversarialFairness)

    mitigator.fit(Cont2d, Cont2d, sensitive_features=Cat)
    assert isinstance(mitigator.backendEngine_.predictor_loss, MSE)
    assert isinstance(mitigator.backendEngine_.adversary_loss, CCE)


@pytest.mark.parametrize("torch3", [True, False])
def test_fake_models(torch3):
    """Test various data types and see if it is interpreted correctly."""
    for ((X, Y, Z), (X_type, Y_type, Z_type)) in generate_data_combinations():
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
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(X.tolist(), Y.tolist(), sensitive_features=Z.tolist())


def test_fake_models_df_inputs():
    """Test model with data frames as input."""
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(
            pd.DataFrame(X), pd.DataFrame(Y), sensitive_features=pd.DataFrame(Z)
        )


def check_type_helper(data, actual_type, valid_choices, invalid_choices):
    """Help to check if distribution types are interpreted correctly."""
    for valid_choice in valid_choices:
        prep = FloatTransformer(valid_choice)
        prep.fit(data)
        assert prep.dist_type_ == actual_type

    for invalid_choice in invalid_choices:
        with pytest.raises(ValueError) as exc:
            prep = FloatTransformer(invalid_choice)
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
        [Keyword_AUTO, Keyword_CONTINUOUS],
        [
            Keyword_BINARY,
            Keyword_CATEGORY,
            Keyword_CLASSIFICATION,
            None,
            "bogus",
        ],
    )
    check_type_helper(
        Cont2d,
        Keyword_CONTINUOUS,
        [Keyword_AUTO, Keyword_CONTINUOUS],
        [
            Keyword_BINARY,
            Keyword_CATEGORY,
            Keyword_CLASSIFICATION,
            None,
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
        prep = FloatTransformer(Keyword_AUTO)
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
    notCat[
        0, 2:
    ] = 0.0  # Special case because now first row sums to one but is not one-hot
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
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X, Y, Z)
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_list_inputs():
    """Test if validate_data properly preprocesses list datasets to ndarray."""
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X.tolist(), Y.tolist(), Z.tolist())
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_df_inputs():
    """Test if validate_data properly preprocesses dataframes to ndarray."""
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(
            pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(Z)
        )
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_ambiguous_rows():
    """Test if an ambiguous number of rows are caught."""
    for ((X, Y, Z), types) in generate_data_combinations():
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
