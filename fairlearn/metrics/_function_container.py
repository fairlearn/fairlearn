# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_NAME = 'metric'

_METRIC_FUNCTION_NONE = "Found 'None' instead of metric function"
_METRIC_FUNCTION_NOT_CALLABLE = "Object passed as metric function not callable"
_SAMPLE_PARAMS_NOT_DICT = "Sample parameters must be a dictionary"


class FunctionContainer:
    """A helper class for metrics.

    Parameters
    ----------
    func : Callable
        The metric function

    name : str
        The name of the metric. If ``None`` then the ``__name__``
        property of the ``func`` is used, or if that is not available
        a default is used.

    sample_params : dict[str,array_like]
        Sample parameters, which are to be sliced up along with
        ``y_true`` and ``y_pred``
    """

    def __init__(self,
                 func: Callable,
                 name: Optional[str],
                 sample_params: Optional[Dict[str, Any]]):
        """Read a placeholder comment."""
        if func is None:
            raise ValueError(_METRIC_FUNCTION_NONE)
        if not callable(func):
            raise ValueError(_METRIC_FUNCTION_NOT_CALLABLE)
        self._func = func

        if name is None:
            if hasattr(func, '__name__'):
                self._name = func.__name__
            else:
                logger.warning("Supplied 'func' had no __name__ attribute")
                self._name = _DEFAULT_NAME
        else:
            self._name = name

        self._sample_params = dict()
        if sample_params is not None:
            if not isinstance(sample_params, dict):
                raise ValueError(_SAMPLE_PARAMS_NOT_DICT)
            for k, v in sample_params.items():
                if v is not None:
                    # Coerce any sample_params to being ndarrays for easy masking
                    self._sample_params[k] = np.asarray(v)

    @property
    def func_(self) -> Callable:
        """Return the contained metric function."""
        return self._func

    @property
    def name_(self) -> str:
        """Return the name of the metric."""
        return self._name

    @property
    def sample_params_(self) -> Dict[str, np.ndarray]:
        """Return the dictionary of sample parameters (as ndarray)."""
        return self._sample_params

    def generate_sample_params_for_mask(self,
                                        mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the sample parameters selected by the given mask."""
        curr_sample_params = dict()
        for name, value in self.sample_params_.items():
            curr_sample_params[name] = value[mask]
        return curr_sample_params

    def evaluate(self,
                 y_true,
                 y_pred,
                 mask: np.ndarray) -> Any:
        """Evaluate the metric for the given mask and input data.

        The mask will be applied to ``y_true``, ``y_pred`` and
        the sample parameters.
        """
        # Following are internal sanity checks
        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_true) == len(y_pred)
        assert len(y_true) == len(mask)
        params = self.generate_sample_params_for_mask(mask)

        return self.func_(y_true[mask], y_pred[mask], **params)

    def evaluate_all(self,
                     y_true,
                     y_pred) -> Any:
        """Evaluate the metric on all data."""
        return self.func_(y_true, y_pred, **self.sample_params_)
