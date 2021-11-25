# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import _IMPORT_ERROR_MESSAGE


# TODO final activation different
# TODO move to backend engines?

def getTorchModel(list_nodes, activation="sigmoid"):
    """Fully connected neural network as a Torch model.

    Parameters
    ----------
    list_nodes: List[int]
        Number of nodes per layer.
    """
    try:
        from torch.nn import Module, Linear, Sigmoid, ModuleList
        from torch.nn.init import xavier_uniform_
    except ImportError:
        raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("torch"))

    class FullyConnected(Module):
        """Fully connected neural network as a PyTorch Module.

        Parameters
        ----------
        list_nodes: List[int]
            Number of nodes per layer.
        """

        def __init__(self, list_nodes, activation):
            """Initialize the layers of the NN."""
            super(FullyConnected, self).__init__()
            layers = []
            for i in range(len(list_nodes) - 1):
                layers.append(Linear(list_nodes[i], list_nodes[i + 1]))
                layers.append(activation)
            # layers.pop(-1)
            self.layers_ = ModuleList(layers)

        def forward(self, x):
            """Propagate x through the network."""
            for layer in self.layers_:
                x = layer(x)
            return x

    if (not activation) or activation == 'sigmoid':
        activation = Sigmoid()
    model = FullyConnected(list_nodes, activation)

    def init_weights(m):
        if isinstance(m, Linear):
            xavier_uniform_(m.weight)
            m.bias.data.fill_(0.)
    model.apply(init_weights)

    return model


def getTensorflowModel(list_nodes, activation='sigmoid'):
    """Fully connected neural network as a Tensorflow model.

    Parameters
    ----------
    list_nodes: List[int]
        Number of nodes per layer.
    """
    try:
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.initializers import GlorotNormal
    except ImportError:
        raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("tensorflow"))

    initializer_w = GlorotNormal(seed=123)

    class FullyConnected(Model):
        """Fully connected neural network as a Tensorflow model.

        Parameters
        ----------
        list_nodes: List[int]
            Number of nodes per layer.
        """

        def __init__(self, list_nodes, activation=activation):
            """Initialize the layers of the NN."""
            super(FullyConnected, self).__init__()
            layers = []
            for i in range(1, len(list_nodes)):
                layers.append(Dense(
                    units=list_nodes[i],
                    kernel_initializer=initializer_w,
                    bias_initializer='zeros',
                    activation=activation,
                ))
            layers.append(Dense(
                list_nodes[-1],
                kernel_initializer=initializer_w,
                bias_initializer='zeros'
            ))
            self.layers_ = layers

        def call(self, x):
            """Propagate x through the network."""
            for layer in self.layers_:
                x = layer(x)
            return x

    model = FullyConnected(list_nodes, activation)

    return model
