# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import (
    _TYPE_COMPLIANCE_ERROR,
    _TYPE_CHECK_ERROR,
    _ARG_ERROR_MESSAGE,
    _TYPE_UNKNOWN_ERROR,
)
from sklearn.utils import check_array
from sklearn.preprocessing import OneHotEncoder
from pandas import Series, DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from numpy import all as np_all
from numpy import sum as np_sum


class FloatTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that maps dataframes to numpy arrays of floats.

    Applies one-hot-encoding to categorical columns, 'passthrough' to
    numerical columns.
    """

    def __init__(self, dist_assumption="auto"):
        """
        Initialize empty transformers with the given distribution assumption.

        Parameters
        ----------
        dist_assumption : str, default = "auto"
            This is a string that indicates an assumption about the
            distribution of
            the data that will be transformed. Possible assumptions are
            "binary", "category", "continuous", "classification",
            "auto" (default).
            If the data is not
            describable using one of these keywords,
            you must build your own transformer instead.
        """
        self.dist_assumption = dist_assumption

    def _prep(self, X, dtype=None, init=False):
        """
        Prepare X by doing some checks and converting to ndarray.

        dtype : numpy.dtype
            None to keep dtypes, float to coerce to numeric.

        init : bool
            Whether this is the first call to _prep or not.

        Returns
        -------
        X : pandas.DataFrame, numpy.ndarray
            validated input
        """
        if init:
            self.in_type_ = type(X)
            self.inferred_type_ = _infer_type(X)
            self.dist_type_ = _get_type(X, self.dist_assumption)
        else:
            if not isinstance(X, self.in_type_):
                raise ValueError(
                    _ARG_ERROR_MESSAGE.format("X", "of type " + self.in_type_)
                )
            inferred = _infer_type(X)
            if not inferred == self.inferred_type_:
                raise ValueError(
                    "Inferred distribution type of X does not match "
                    + self.inferred_type_
                )
            if not _get_type(X, self.dist_assumption) == self.dist_type_:
                raise ValueError(_TYPE_CHECK_ERROR.format(self.dist_type_))

        if self.in_type_ == DataFrame:
            self.columns_ = X.columns

        X = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=dtype,
            ensure_2d=False,
        )

        self.input_dim_ = X.ndim
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    def fit(self, X, y=None):
        """Fit the three transformers."""
        X = self._prep(X, init=True)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1]

        if self.inferred_type_ in ["binary", "multiclass"]:
            # NOTE: if 'binary' then it could be possible it is already 0/1
            # encoded. So may want to skip redundant OHE in that case...
            self.ct_ = OneHotEncoder(
                drop="if_binary", sparse=False, handle_unknown="error"
            )
            self.ct_.fit(X)
            self.n_features_out_ = sum(
                len(cat) if len(cat) != 2 else 1 for cat in self.ct_.categories_
            )
        # elif "multilabel-indicator" needn't be encoded, so we do not create
        # an encoder then.

        return self

    def transform(self, X):
        """Transform X using the fitted encoder or passthrough."""
        if self.inferred_type_ in ["continuous", "continuous-multioutput"]:
            return self._prep(X, dtype=float)
        elif self.inferred_type_ in ["binary", "multiclass"]:
            return self.ct_.transform(self._prep(X)).astype(float)
        elif self.inferred_type_ == "multilabel-indicator":
            return self._prep(X, dtype=float)

    def inverse_transform(self, y):
        """Transform y back to X using the inverse transform of the encoder."""
        inverse = None
        if self.inferred_type_ in ["continuous", "continuous-multioutput"]:
            inverse = y
        elif self.inferred_type_ in ["binary", "multiclass"]:
            inverse = self.ct_.inverse_transform(y)
        elif self.inferred_type_ == "multilabel-indicator":
            inverse = y

        if self.input_dim_ == 1:
            inverse = inverse.reshape(-1)

        # Because we are kind, we try to translate back to the original data
        # type, but we only support DataFrame, Series, list(, ndarray).
        if self.in_type_ == DataFrame:
            inverse = DataFrame(inverse, columns=self.columns_)
        elif self.in_type_ == Series:
            inverse = Series(inverse)
        elif self.in_type_ == list:
            inverse = inverse.tolist()

        return inverse

# hash = lambda d : id(d) + sum(d.shape) + d.size + id(d.dtype)
# memoize_store = {}
def _infer_type(data):
    """
    Memoize type_of_target without prohibiting data garbage-disposal.
    """
    # memoized = memoize_store.get(hash(data), None)
    # if memoized is not None:
    #     print("hit")
    #     return memoized
    # print("miss")
    inferred = type_of_target(data)
    # memoize_store[hash(data)] = inferred
    return inferred


def _get_type(data, assumption):
    """Get the type (binary, category, continuous) of the data under assump."""
    inferred = _infer_type(data)
    if inferred == "multilabel-indicator":
        #  Design choice: multiple binary columns are not supported.
        #                 multiple columns may only be one-hot encoding
        # FIXME provide warning and switch to binary2d
        # or not? because it could be multiple categorical... who knows!
        if not np_all(np_sum(data, axis=1) == 1):
            raise ValueError(_TYPE_CHECK_ERROR.format("category"))

    # Match inferred with dist_assumption
    if inferred == "binary" and assumption in [
        "binary",
        "classification",
        "auto",
    ]:
        return "binary"
    elif inferred in ["multiclass", "multilabel-indicator"] and assumption in [
        "category",
        "classification",
        "auto",
    ]:
        return "category"
    elif inferred in [
        "continuous",
        "continuous-multioutput",
    ] and assumption in [
        "continuous",
        "auto",
    ]:
        return "continuous"
    elif inferred == "multiclass-multioutput":
        raise NotImplementedError("multiclass-multioutput not supported")
        # NOTE we can actually implement this as concatenations of loss
        # functions, but that is up to the future user to decide for now.
    elif inferred == "unknown":
        raise ValueError(_TYPE_UNKNOWN_ERROR)

    raise ValueError(_TYPE_COMPLIANCE_ERROR.format(assumption, inferred))
