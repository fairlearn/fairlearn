# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import _IMPORT_ERROR_MESSAGE, _KWARG_ERROR_MESSAGE

class AdversarialDebiasing():
    def __init__(self, *, 
            environment = 'torch', 
            predictor_model, 
            adversary_model,
    ):
        if environment == 'torch':
            self.torch = True
            try:
                import torch
            except ImportError:
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format('torch'))
        elif environment == 'tensorflow':
            self.tensorflow = True
        else:
            raise ValueError(_KWARG_ERROR_MESSAGE.format("environment", "one of \[\'torch\',\'tensorflow\'\]"))


    def train(X, Y, Z, epochs, batch_size)

    def _train_torch():
        if not self.torch:
            raise RuntimeError(_IMPORT_ERROR_MESSAGE.format('torch')) 
        