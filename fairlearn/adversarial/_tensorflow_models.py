# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


class regressor(Model):
    def __init__(self, N_in, N_out):
        super(regressor, self).__init__()
        self.layer = Dense(N_out)

    def call(self, x):
        x = self.layer(x)
        return x
