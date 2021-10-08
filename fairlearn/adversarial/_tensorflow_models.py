# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


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
