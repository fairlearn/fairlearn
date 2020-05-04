# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

class ConstantPredictor:
    def __init__(self, constant_value):
        self._constant_value = constant_value

    def predict(self, *args, **kwargs):
        return self._constant_value