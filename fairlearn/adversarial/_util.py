from ._constants import (
    _KWARG_ERROR_MESSAGE,
    _TYPE_CHECK_ERROR,
    _TYPE_COMPLIANCE_ERROR,
    _INVALID_DATA,
    _PREDICTION_FUNCTION_AMBIGUOUS,
)
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from numpy import zeros, argmax, logical_or, arange
from numpy import all as np_all
from numpy import sum as np_sum
from enum import Enum


class Keyword(Enum):
    """Keywords to use to define template loss/predictor functions in AdversarialFairness."""

    AUTO = "auto"
    CLASSIFICATION = "classification"
    BINARY = "binary"
    CATEGORY = "category"
    CONTINUOUS = "continuous"


def _check_array(X):
    """
    Validate the input array, and possible coerce to 2D.

    Calls :code:`sklearn.utils.check_array` on parameter X with the
    parameters suited for Adversarial Mitigation.
    """
    return check_array(
        X,
        accept_sparse=False,
        accept_large_sparse=False,
        dtype=float,
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=True,
        allow_nd=False,
        ensure_min_samples=1,
        ensure_min_features=1,
        estimator=None,
    )


def interpret_keyword(data, input_type, param_name, data_name):
    if callable(input_type):
        return input_type
    elif isinstance(input_type, str):
        if input_type.lower() in [kw.value for kw in Keyword]:
            # Translate string to enum.
            for kw in Keyword:
                if input_type.lower() == kw.value:
                    input_type = kw
                    break
            if isinstance(input_type, Keyword):
                return check_and_infer_type(data, input_type, data_name)
    raise ValueError(
        _KWARG_ERROR_MESSAGE.format(
            param_name,
            "one of {} or a callable".format([kw.value for kw in Keyword]),
        )
    )


def check_and_infer_type(Y, keyword, data_name):
    """Identify user query :code:`choice`."""
    inferred = type_of_target(Y)

    if inferred in ["multiclass", "multiclass-multioutput", "unknown"]:
        # We do not support these types! Multiclass should be one-hot-encoded.
        raise ValueError(_INVALID_DATA.format(data_name))
    if inferred in ["binary", "multilabel-indicator"]:
        # We need discrete values to be 0 or 1 in training neural networks,
        # because the losses expect this!
        if not np_all(logical_or(Y == 0, Y == 1)):
            raise ValueError(_INVALID_DATA.format(data_name))
    if inferred == "multilabel-indicator":
        #  Design choice: multiple binary columns are not supported.
        #                 multiple columns may only be one-hot encoding
        if not np_all(np_sum(Y, axis=1) == 1):
            raise ValueError(
                _TYPE_CHECK_ERROR.format(data_name, Keyword.CATEGORY.value)
            )

    # Match inferred with keyword
    if inferred == "binary" and keyword in [
        Keyword.BINARY,
        Keyword.CLASSIFICATION,
        Keyword.AUTO,
    ]:
        return Keyword.BINARY
    elif inferred == "multilabel-indicator" and keyword in [
        Keyword.CATEGORY,
        Keyword.CLASSIFICATION,
        Keyword.AUTO,
    ]:
        return Keyword.CATEGORY
    elif inferred in ["continuous", "continuous-multioutput"] and keyword in [
        Keyword.CONTINUOUS,
        Keyword.AUTO,
    ]:
        return Keyword.CONTINUOUS

    raise ValueError(
        _TYPE_COMPLIANCE_ERROR.format(data_name, keyword.value, inferred)
    )


def _get_function(keyword):
    """
    Infer prediction function.

    If
    """
    if keyword == Keyword.BINARY:
        return lambda pred: (pred >= 0.5).astype(float)
    elif keyword == Keyword.CATEGORY:

        def loss(pred):
            shape = pred.shape
            c = argmax(pred, axis=1)
            b = zeros(shape, dtype=float)
            a = arange(shape[0])
            b[a, c] = 1
            return b

        return loss
    elif keyword == Keyword.CONTINUOUS:
        return lambda pred: pred
    elif callable(keyword):
        return keyword
    else:
        raise ValueError(
            _PREDICTION_FUNCTION_AMBIGUOUS.format(
                keyword, ["binary", "category", "continuous"]
            )
        )
