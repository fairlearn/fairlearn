# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from torch.nn import Module, Linear, Sigmoid, ModuleList


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
