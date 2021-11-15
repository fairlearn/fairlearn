# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import sys  # , fake_torch
import pytest
import numpy as np
import pandas as pd
from fairlearn.adversarial import Keyword, AdversarialFairness, \
    AdversarialFairnessClassifier, AdversarialFairnessRegressor
from fairlearn.adversarial._constants import _TYPE_CHECK_ERROR


model_class = type('Model', (object,), {})


class loss_class:
    def __init__(self, **kwargs): pass
    # class loss:
    #     def backward(self, **kwargs): return
    def __call__(self, x, y): return  # self.loss()


BCE = type('binary', (loss_class,), {})
CCE = type('category', (loss_class,), {})
MSE = type('continuous', (loss_class,), {})


def KeywordToClass(kw):
    print(kw)
    index = [Keyword.BINARY, Keyword.CATEGORY, Keyword.CONTINUOUS].index(kw)
    print(index)
    print([BCE, CCE, MSE][index])
    return [BCE, CCE, MSE][index]


layer_class = type('Layer', (object,), {})


class fake_torch:
    class nn:
        Module = type('Module', (model_class,), {
            'forward': lambda self, x: x,
            'parameters': lambda self: (i for i in [1, 2]),
            'train': lambda self: None,
            '__call__': lambda self, X: self.forward(X)})
        BCEWithLogitsLoss = type('BCEWithLogitsLoss', (BCE,), {})
        CrossEntropyLoss = type('CrossEntropyLoss', (CCE,), {})
        MSELoss = type('MSELoss', (MSE,), {})

    class optim:
        class Adam:
            def __init__(self, params, **kwargs): pass
            def zero_grad(self): pass

    class from_numpy:
        def __init__(self, X): self.X = X

        def float(self): return self.X
    # def clone(self, X): return X.copy()


class fake_tensorflow:
    class keras:
        Model = type('Model', (model_class,), {})

        class losses:
            BinaryCrossentropy = type('BinaryCrossentropy', (BCE,), {})
            CategoricalCrossentropy = type('CategoricalCrossentropy', (CCE,), {})
            MeanSquaredError = type('MeanSquaredError', (MSE,), {})

        class optimizers:
            class Adam:
                def __init__(self, **kwargs): pass


class RemoveAll():
    def _evaluate(self, X): return X
    def _train_step(self, X, Y, Z): return (0, 0)
    def _setup_optimizer(self, optimizer): pass
    def _get_loss(self, Y, choice, data_name): pass


class RemoveTrainStep():
    def _train_step(self, X, Y, Z): return (0, 0)


rows = 60
cols = 5
Bin2d = np.random.choice([0., 1.], size=(rows, cols))
Bin1d = np.random.choice([0., 1.], size=(rows, 1))
Cat = np.zeros((rows, cols), dtype=float)
Cat[np.arange(rows), np.random.choice([i for i in range(cols)], size=(rows,))] = 1.
Cont2d = np.random.rand(rows, cols)
Cont1d = np.random.rand(rows, 1)


def generate_data_combinations(n=10):
    datas = [Bin1d, Cat, Cont2d, Cont1d]
    dist_type = [Keyword.BINARY, Keyword.CATEGORY, Keyword.CONTINUOUS, Keyword.CONTINUOUS]
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


def get_instance(
        cls=AdversarialFairness,
        fake_mixin=False,
        fake_training=False,
        torch=True,
        tensorflow=False):
    if torch:
        sys.modules['torch'] = fake_torch
    else:
        sys.modules['torch'] = None
    if tensorflow:
        sys.modules['tensorflow'] = fake_tensorflow
    else:
        sys.modules['tensorflow'] = None

    if torch:
        predictor = fake_torch.nn.Module()
        adversary = fake_torch.nn.Module()
    elif tensorflow:
        predictor = fake_tensorflow.keras.Model()
        adversary = fake_tensorflow.keras.Model()

    mitigator = cls(
        predictor_model=predictor,
        adversary_model=adversary
    )
    if fake_mixin:
        mitigator._extend_instance(RemoveAll)
    else:
        if fake_training:
            mitigator._extend_instance(RemoveTrainStep)
    return mitigator

# CURRENTLY not testing what the models look like, just that it is correct type


@pytest.mark.parametrize("torch", [True, False])
def test_classifier(torch):
    mitigator = get_instance(AdversarialFairnessClassifier, fake_training=True,
                             torch=torch, tensorflow=not torch)
    assert isinstance(mitigator, AdversarialFairness)

    mitigator.fit(Cont2d, Bin1d, sensitive_features=Cat)
    assert isinstance(mitigator.predictor_loss, BCE)
    assert isinstance(mitigator.adversary_loss, CCE)


@pytest.mark.parametrize("torch", [True, False])
def test_regressor(torch):
    mitigator = get_instance(AdversarialFairnessRegressor, fake_training=True,
                             torch=torch, tensorflow=not torch)
    assert isinstance(mitigator, AdversarialFairness)

    mitigator.fit(Cont2d, Cont2d, sensitive_features=Cat)
    assert isinstance(mitigator.predictor_loss, MSE)
    assert isinstance(mitigator.adversary_loss, CCE)


@pytest.mark.parametrize("torch", [True, False])
def test_fake_models(torch):
    for ((X, Y, Z), (X_type, Y_type, Z_type)) in generate_data_combinations():
        mitigator = get_instance(fake_training=True,
                                 torch=torch, tensorflow=not torch)
        mitigator.fit(X, Y, sensitive_features=Z)
        assert isinstance(mitigator.predictor_loss, KeywordToClass(Y_type))
        assert isinstance(mitigator.adversary_loss, KeywordToClass(Z_type))


def test_fake_models_list_inputs():
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(X.tolist(), Y.tolist(), sensitive_features=Z.tolist())


def test_fake_models_df_inputs():
    for ((X, Y, Z), types) in generate_data_combinations():
        mitigator = get_instance(fake_mixin=True)
        mitigator.fit(pd.DataFrame(X), pd.DataFrame(Y), sensitive_features=pd.DataFrame(Z))


def check_type_helper(data, actual_type, valid_choices, invalid_choices):
    data_name = "test"
    mitigator = get_instance()
    for valid_choice in valid_choices:
        assert mitigator._check_type(data, valid_choice, data_name) == actual_type

    for invalid_choice in invalid_choices:
        with pytest.raises(ValueError) as exc:
            assert not mitigator._check_type(data, invalid_choice, data_name)
            assert str(exc.value) == _TYPE_CHECK_ERROR.format(data_name, invalid_choice)


def test_check_type_correct_data():
    check_type_helper(Bin1d, Keyword.BINARY,
                      [Keyword.AUTO, Keyword.BINARY, Keyword.CLASSIFICATION],
                      [Keyword.CATEGORY, None, 'bogus'])
    check_type_helper(Cat, Keyword.CATEGORY,
                      [Keyword.AUTO, Keyword.CATEGORY, Keyword.CLASSIFICATION],
                      [Keyword.BINARY, None, 'bogus'])
    check_type_helper(Cont1d, Keyword.CONTINUOUS,
                      [Keyword.AUTO, Keyword.CONTINUOUS],
                      [Keyword.BINARY, Keyword.CATEGORY, Keyword.CLASSIFICATION, None, 'bogus'])
    check_type_helper(Cont2d, Keyword.CONTINUOUS,
                      [Keyword.AUTO, Keyword.CONTINUOUS],
                      [Keyword.BINARY, Keyword.CATEGORY, Keyword.CLASSIFICATION, None, 'bogus'])
    # Bin2d is not interpretable as binary 2d, need to set custom loss for this.
    check_type_helper(Bin2d, Keyword.CONTINUOUS,
                      [Keyword.CONTINUOUS],
                      [Keyword.BINARY, Keyword.CATEGORY, Keyword.CLASSIFICATION, None, 'bogus'])

    data_name = "test"
    mitigator = get_instance()
    with pytest.raises(ValueError) as exc:
        assert mitigator._check_type(Bin2d, Keyword.AUTO, data_name) is None
        assert str(exc.value) == _TYPE_CHECK_ERROR.format(data_name, Keyword.AUTO)


def test_check_type_faulty_data():
    notBin1d = Bin1d.copy()
    notBin1d[0] = 0.1
    check_type_helper(notBin1d, Keyword.CONTINUOUS,
                      [Keyword.AUTO, Keyword.CONTINUOUS],
                      [Keyword.BINARY, Keyword.CATEGORY, Keyword.CLASSIFICATION])
    notCat = Cat.copy()
    notCat[0, 0:2] = 1
    check_type_helper(notCat, Keyword.CONTINUOUS,
                      [Keyword.CONTINUOUS],  # Auto doesnt work on ambiguous
                      [Keyword.BINARY, Keyword.CATEGORY, Keyword.CLASSIFICATION])
    notCat[0, 0] = 0.5
    check_type_helper(notCat, Keyword.CONTINUOUS,
                      [Keyword.AUTO, Keyword.CONTINUOUS],  # Auto does work here
                      [Keyword.BINARY, Keyword.CATEGORY, Keyword.CLASSIFICATION])
    notCat[0, 1] = 0.5
    notCat[0, 2:] = 0.  # Special case because now first row sums to one but is not one-hot
    check_type_helper(notCat, Keyword.CONTINUOUS,
                      [Keyword.AUTO, Keyword.CONTINUOUS],  # Auto does work here
                      [Keyword.BINARY, Keyword.CATEGORY, Keyword.CLASSIFICATION])


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
        X, Y, Z = mitigator._validate_input(pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(Z))
        for x in (X, Y, Z):
            check_2dnp(x)


def test_validate_data_ambiguous_rows():
    for ((X, Y, Z), types) in generate_data_combinations():
        X = X[:5, :]
        mitigator = get_instance(fake_mixin=True)
        with pytest.raises(ValueError) as exc:
            mitigator._validate_input(X, Y, Z)
            assert str(exc.value) == \
                "Input data has an ambiguous number of rows: {}, {}, {}.".format(
                    X.shape[0], Y.shape[0], Z.shape[0])


sys.modules['torch'] = None
sys.modules['tensorflow'] = None
