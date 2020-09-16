# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np


class FunctionContainer:
    def __init__(self, func, name, sample_param_names, params):
        self._func = func
        if name is None:
            self._name = func.__name__
        else:
            self._name = name

        assert isinstance(sample_param_names, list)
        self._sample_param_names = sample_param_names
        assert isinstance(params, dict)
        self._params = params

        # Coerce any sample_params to being ndarrays for easy masking
        for param_name in self.sample_param_names:
            assert param_name in self.params
            self.params[param_name] = np.asarray(self.params[param_name])

    @property
    def func(self):
        return self._func

    @property
    def name(self):
        return self._name

    @property
    def sample_param_names(self):
        return self._sample_param_names

    @property
    def params(self):
        return self._params

    def generate_params_for_mask(self, mask):
        curr_params = dict()
        for name, value in self.params.items():
            if name in self.sample_param_names:
                # Constructor has forced the value to be an ndarray
                curr_params[name] = value[mask]
            else:
                curr_params[name] = value
