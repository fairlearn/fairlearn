# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import math
import numpy as np
import sklearn.metrics as skm

from ._input_manipulations import _convert_to_ndarray_and_squeeze

_Y_TRUE_NOT_0_1 = "Only 0 and 1 are allowed in y_true and both must be present"


def _balanced_root_mean_squared_error(y_true, y_pred, sample_weight=None):
    r"""Calculate the mean of the root mean squared error (RMSE) for the positive and negative cases.

    Used for binary logistic regression, this computes the error as

    .. math::
       \frac{\text{RMSE}(Y=0) + \text{RMSE}(Y=1)}{2}

    The classes are constrained to be :math:`\in {0, 1}`. The :code:`y_true` values must
    always be one of these, while :code:`y_pred` can be a continuous probability
    (which could be thresholded to get a predicted class).

    Internally, this builds on the
    :py:func:`sklearn.metrics.mean_squared_error` routine.
    """
    y_ta = _convert_to_ndarray_and_squeeze(y_true)
    y_pa = _convert_to_ndarray_and_squeeze(y_pred)
    s_w = np.ones(len(y_ta))
    if sample_weight is not None:
        s_w = _convert_to_ndarray_and_squeeze(sample_weight)

    y_ta_values = np.unique(y_ta)
    if not np.array_equal(y_ta_values, [0, 1]):
        raise ValueError(_Y_TRUE_NOT_0_1)

    errs = np.zeros(2)
    for i in range(0, 2):
        indices = (y_ta == i)
        y_ta_s = y_ta[indices]
        y_pa_s = y_pa[indices]
        s_w_s = s_w[indices]
        errs[i] = math.sqrt(skm.mean_squared_error(y_ta_s, y_pa_s, sample_weight=s_w_s))

    return errs.mean()
