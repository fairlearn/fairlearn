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

from fairlearn.adversarial._adversarial_mitigation import (
    _AdversarialFairness,  # We just test the base class because this covers all
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

    def randperm(X):
        return np.random.choice(range(X), size=X, replace=False)

    class nn:
        """Mock of torch.nn."""

        class Module(model_class):  # noqa: D106
            w = 0

            def forward(self, x):
                return x

            def parameters(self):
                return self.w

            def train(self):
                return None

            def __call__(self, X):
                return self.forward(X)

            def apply(self, weights):
                self.w = weights

        BCELoss = type("BCELoss", (BCE,), {})
        CrossEntropyLoss = type("CrossEntropyLoss", (CCE,), {})
        MSELoss = type("MSELoss", (MSE,), {})

        class Linear:  # noqa: D106
            def __init__(self, a, b):
                self.a = a
                self.b = b

        def ReLU():
            return type("ReLU", (), {"__call__": lambda x: np.max(x, 0)})  # noqa: E731

        def LeakyReLU():
            return type("LeakyReLU", (), {"__call__": lambda x: np.max(x, 0.1 * x)})

        Sigmoid = lambda: type("Sigmoid", (), {"__call__": lambda x: x})  # noqa: E731
        Softmax = lambda: type("Softmax", (), {"__call__": lambda x: x})  # noqa: E731

        class ModuleList:  # noqa: D106
            def __init__(self, layers):
                self.layers = layers

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

        class activations:  # noqa: D106
            def deserialize(item):
                return type(item, (), {"__call__": lambda x: x})

        class layers:  # noqa: D106
            class Dense:  # noqa: D106
                def __init__(self, units, kernel_initializer, bias_initializer):
                    self.b = units

        class initializers:  # noqa: D106
            class GlorotNormal:  # noqa: D106
                pass

        Model = type("Model", (model_class,), {})

        class losses:
            """Mock of tf.keras.losses."""

            BinaryCrossentropy = type("BinaryCrossentropy", (BCE,), {})
            CategoricalCrossentropy = type("CategoricalCrossentropy", (CCE,), {})
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
Cat[np.arange(rows), np.random.choice([i for i in range(cols)], size=(rows,))] = 1.0
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
    total_combinations = K**3
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
        cols = self.base._y_transform.n_features_out_
        rows = len(X)
        y = []
        for row in range(rows):
            rng = np.random.default_rng(int(np.round(np.mean(X[row]) * (2**32))))
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

    def shuffle(self, X, Y, A):
        return X, Y, A

    def train_step(self, X, Y, Z):  # noqa: D102
        return (0, 0)


class RemoveTrainStepTensorflow(TensorflowEngine):
    """Mock train_step only, then we can still perform Backend inits."""

    def train_step(self, X, Y, Z):  # noqa: D102
        return (0, 0)


def get_instance(
    cls=_AdversarialFairness,
    fake_mixin=False,
    fake_training=False,
    torch=True,
    tensorflow=False,
    **kwargs,
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

    default_kwargs = dict()

    if torch:
        default_kwargs["predictor_model"] = fake_torch.nn.Module()
        default_kwargs["adversary_model"] = fake_torch.nn.Module()
    elif tensorflow:
        default_kwargs["predictor_model"] = fake_tensorflow.keras.Model()
        default_kwargs["adversary_model"] = fake_tensorflow.keras.Model()

    default_kwargs.update(kwargs)
    mitigator = cls(**default_kwargs)

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
