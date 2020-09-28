# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np


class FunctionContainer:
    """Read a placeholder comment."""

    def __init__(self, func, name, sample_params):
        """Read a placeholder comment."""
        assert func is not None
        assert callable(func)
        self._func = func

        if name is None:
            self._name = func.__name__
        else:
            self._name = name

        self._sample_params = dict()
        if sample_params is not None:
            assert isinstance(sample_params, dict)
            self._sample_params = sample_params

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

    def generate_sample_params_for_mask(self, mask):
        """Read a placeholder comment."""
        curr_sample_params = dict()
        for name, value in self.sample_params_.items():
            curr_sample_params[name] = value[mask]
        return curr_sample_params

    def evaluate(self, y_true, y_pred, mask):
        """Read a placeholder comment."""
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_true) == len(y_pred)
        assert len(y_true) == len(mask)
        params = self.generate_sample_params_for_mask(mask)

        return self.func_(y_true[mask], y_pred[mask], **params)

    def evaluate_all(self, y_true, y_pred):
        """Read a placeholder comment."""
        return self.func_(y_true, y_pred, **self.sample_params_)
