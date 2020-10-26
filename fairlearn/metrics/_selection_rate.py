# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
from typing import Any

from fairlearn.metrics._input_manipulations import _convert_to_ndarray_and_squeeze


_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE = "Empty y_pred passed to selection_rate function."


def selection_rate(y_true,
                   y_pred,
                   *,
                   pos_label: Any = 1,
                   sample_weight=None) -> float:
    """Calculate the fraction of predicted labels matching the 'good' outcome.

    The argument `pos_label` specifies the 'good' outcome. For consistency with
    other metric functions, the ``y_true`` argument is required, but ignored.


    Parameters
    ----------
    y_true : array_like
        The true labels (ignored)

    y_pred : array_like
        The predicted labels

    pos_label : Scalar
        The label to treat as the 'good' outcome

    sample_weight : array_like
        Optional array of sample weights
    """
    if len(y_pred) == 0:
        raise ValueError(_EMPTY_INPUT_PREDICTIONS_ERROR_MESSAGE)

    selected = (_convert_to_ndarray_and_squeeze(y_pred) == pos_label)
    s_w = np.ones(len(selected))
    if sample_weight is not None:
        s_w = np.squeeze(np.asarray(sample_weight))

    return np.dot(selected, s_w) / s_w.sum()
