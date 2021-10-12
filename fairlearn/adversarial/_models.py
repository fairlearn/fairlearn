# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import _IMPORT_ERROR_MESSAGE

def getTorchModel():
    try:
        from torch.nn import Module, Linear, Sigmoid, ModuleList
    except ImportError as e:
        raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("torch"))

    class FullyConnected(Module):
        """Fully connected neural network as a PyTorch Module.

        Parameters
        ----------
        list_nodes: List[int]
            Number of nodes per layer.
        """

        def __init__(self, list_nodes):
            """Initialize the layers of the NN."""
            super(FullyConnected, self).__init__()
            layers = []
            for i in range(len(list_nodes) - 1):
                layers.append(Linear(list_nodes[i], list_nodes[i + 1]))
                layers.append(Sigmoid())
            layers.pop(-1)
            self.layers_ = ModuleList(layers)

        def forward(self, x):
            """Propagate x through the network."""
            for layer in self.layers_:
                x = layer(x)
            return x

    return FullyConnected

def getTensorflowModel():
    try:
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense
    except ImportError as e:
        raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("tensorflow"))

    class FullyConnected(Model):
        """Fully connected neural network as a Tensorflow model.

        Parameters
        ----------
        list_nodes: List[int]
            Number of nodes per layer.
        """

        def __init__(self, list_nodes):
            """Initialize the layers of the NN."""
            super(FullyConnected, self).__init__()
            layers = []
            for i in range(1, len(list_nodes) - 1):
                layers.append(Dense(list_nodes[i], activation='sigmoid'))
            layers.append(Dense(list_nodes[-1]))
            self.layers_ = layers

        def call(self, x):
            """Propagate x through the network."""
            for layer in self.layers_:
                x = layer(x)
            return x
    
    return FullyConnected