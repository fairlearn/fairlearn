# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from torch.nn import Module, Linear


class regressor(Module):
    def __init__(self, N_in, N_out):
        super(regressor, self).__init__()
        self.layers = Linear(N_in, N_out)

    def forward(self, x):
        return self.layers(x)
