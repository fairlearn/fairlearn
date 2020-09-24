# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import copy
import numpy as np


class FunctionContainer:
    """Read a placeholder comment."""

    def __init__(self, func, name, sample_params, params):
        """Read a placeholder comment."""
        assert func is not None
        self._func = func

        if name is None:
            self._name = func.__name__
        else:
            self._name = name

        self._sample_params = dict()
        if sample_params is not None:
            assert isinstance(sample_params, dict)
            self._sample_params = sample_params
        self._params = dict()
        if params is not None:
            assert isinstance(params, dict)
            self._params = params

        all_param_names = list(self._sample_params.keys()) + list(self._params.keys())
        unique_param_names = np.unique(all_param_names)
        assert len(all_param_names) == len(unique_param_names)

        # Coerce any sample_params to being ndarrays for easy masking
        for k, v in self._sample_params.items():
            self._sample_params[k] = np.asarray(v)

    @property
    def func_(self):
        """Read a placeholder comment."""
        return self._func

    @property
    def name_(self):
        """Read a placeholder comment."""
        return self._name

    @property
    def sample_params_(self):
        """Read a placeholder comment."""
        return self._sample_params

    @property
    def params_(self):
        """Read a placeholder comment."""
        return self._params

    def generate_params_for_mask(self, mask):
        """Read a placeholder comment."""
        curr_params = copy.deepcopy(self.params_)
        for name, value in self.sample_params_.items():
            curr_params[name] = value[mask]
        return curr_params

    def evaluate(self, y_true, y_pred, mask):
        """Read a placeholder comment."""
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_true) == len(y_pred)
        assert len(y_true) == len(mask)
        params = self.generate_params_for_mask(mask)

        return self.func_(y_true[mask], y_pred[mask], **params)

    def evaluate_all(self, y_true, y_pred):
        """Read a placeholder comment."""
        all_params = {**self.params_, **self.sample_params_}
        return self.func_(y_true, y_pred, **all_params)
