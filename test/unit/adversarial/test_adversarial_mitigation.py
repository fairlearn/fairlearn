# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import sys  # , fake_torch
import pytest
import numpy as np
import pandas as pd
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
    def __init__(self, **kwargs):
        pass

    # class loss:
    #     def backward(self, **kwargs): return
    def __call__(self, x, y):
        return  # self.loss()


BCE = type(Keyword_BINARY, (loss_class,), {})
CCE = type(Keyword_CATEGORY, (loss_class,), {})
MSE = type(Keyword_CONTINUOUS, (loss_class,), {})


def KeywordToClass(kw):
    index = [Keyword_BINARY, Keyword_CATEGORY, Keyword_CONTINUOUS].index(kw)
    return [BCE, CCE, MSE][index]


layer_class = type("Layer", (object,), {})


class fake_torch:
    class nn:
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
        class Adam:
            def __init__(self, params, **kwargs):
                pass

            def zero_grad(self):
                pass

    class from_numpy:
        def __init__(self, X):
            self.X = X

        def float(self):
            return self.X

    # def clone(self, X): return X.copy()


class fake_tensorflow:
    class keras:
        Model = type("Model", (model_class,), {})

        class losses:
            BinaryCrossentropy = type("BinaryCrossentropy", (BCE,), {})
            CategoricalCrossentropy = type(
                "CategoricalCrossentropy", (CCE,), {}
            )
            MeanSquaredError = type("MeanSquaredError", (MSE,), {})

        class optimizers:
            class Adam:
                def __init__(self, **kwargs):
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
    def __init__(self, base, X, Y, Z):
        pass

    def evaluate(self, X):
        return X

    def train_step(self, X, Y, Z):
        return (0, 0)

    def setup_optimizer(self, optimizer):
        pass

    def get_loss(self, Y, choice, data_name):
        pass


class RemoveTrainStepPytorch(PytorchEngine):
    def train_step(self, X, Y, Z):
        return (0, 0)


class RemoveTrainStepTensorflow(TensorflowEngine):
    def train_step(self, X, Y, Z):
        return (0, 0)


def get_instance(
    cls=AdversarialFairness,
    fake_mixin=False,
    fake_training=False,
    torch=True,
    tensorflow=False,
):
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


# CURRENTLY not testing what the models look like, just that it is correct type


@pytest.mark.parametrize("torch1", [True, False])
def test_classifier(torch1):
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
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(X.tolist(), Y.tolist(), sensitive_features=Z.tolist())


def test_fake_models_df_inputs():
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(
            pd.DataFrame(X), pd.DataFrame(Y), sensitive_features=pd.DataFrame(Z)
        )


def check_type_helper(data, actual_type, valid_choices, invalid_choices):
    for valid_choice in valid_choices:
        prep = FloatTransformer(valid_choice)
        prep.fit(data)
        assert (
            prep.dist_type_ == actual_type
        )

    for invalid_choice in invalid_choices:
        with pytest.raises(ValueError) as exc:
            prep = FloatTransformer(invalid_choice)
            prep.fit(data)
            assert str(exc.value) == _TYPE_COMPLIANCE_ERROR.format(invalid_choice, prep.inferred_type_)


def test_check_type_correct_data():
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
        assert str(exc.value) == _TYPE_COMPLIANCE_ERROR.format(Keyword_AUTO, prep.inferred_type_)


def test_check_type_faulty_data():
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
    assert isinstance(X, np.ndarray)
    assert X.ndim == 2
    assert len(X.shape) == 2
    assert X.dtype == np.float


def test_validate_data():
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X, Y, Z)
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_list_inputs():
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(X.tolist(), Y.tolist(), Z.tolist())
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_df_inputs():
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        X, Y, Z = mitigator._validate_input(
            pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(Z)
        )
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_ambiguous_rows():
    for ((X, Y, Z), types) in generate_data_combinations():
        X = X[:5, :]
        mitigator = get_instance(fake_mixin=True)
        with pytest.raises(ValueError) as exc:
            mitigator._validate_input(X, Y, Z)
            assert str(
                exc.value
            ) == "Input data has an ambiguous number of rows: {}, {}, {}.".format(
                X.shape[0], Y.shape[0], Z.shape[0]
            )


sys.modules["torch"] = None
sys.modules["tensorflow"] = None
