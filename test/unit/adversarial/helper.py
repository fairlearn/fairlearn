# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
This file contains the helper functions for test_adversarial_mitigation.py.

To summarize, we create *mock* Torch and TensorFlow modules. They have
no functionality, but merely serve as an expected interface that our code
should adhere to. Breaking changes in Torch/TF API should be manually reflected
here. Additionally, we generate data here.
"""

import sys
import numpy as np

from fairlearn.adversarial import (
    AdversarialFairness,
)
from fairlearn.adversarial._pytorch_engine import PytorchEngine
from fairlearn.adversarial._tensorflow_engine import TensorflowEngine
from fairlearn.adversarial._backend_engine import BackendEngine


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
            """Mock of base optimizer, with empty init and zerograd."""

            def __init__(self, params, **kwargs):
                """Mock init because this mock optimizer is initialized."""
                pass

            def zero_grad(self):
                """Mock zerograd because this mock optimizer is used."""
                pass

        class Adam(Optimizer):
            """Mock of pytorch Adam optimizer, with empty init and zerograd."""

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
                """Mock of base optimizer."""

                def __init__(self, **kwargs):  # noqa: D107
                    pass

            class Adam(Optimizer):
                """Mock of pytorch Adam optimizer."""

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


sys.modules["torch"] = None
sys.modules["tensorflow"] = None
