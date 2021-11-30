# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import (
    _TYPE_COMPLIANCE_ERROR,
    _TYPE_CHECK_ERROR,
    _ARG_ERROR_MESSAGE,
)
from sklearn.utils import check_array
from sklearn.preprocessing import OneHotEncoder
from pandas import Series, DataFrame
from sklearn.base import TransformerMixin
from sklearn.utils.multiclass import type_of_target
from numpy import all as np_all
from numpy import sum as np_sum


class FloatTransformer(TransformerMixin):
    """
    Transformer that maps dataframes to numpy arrays of floats.

    Applies one-hot-encoding to categorical columns, 'passthrough' to
    numerical columns.
    """

    def __init__(self, dist_assumption="auto"):
        """Initialize empty transformers."""
        self.dist_assumption = dist_assumption

    def _get_type(self, data, inferred):
        if inferred == "multilabel-indicator":
            #  Design choice: multiple binary columns are not supported.
            #                 multiple columns may only be one-hot encoding
            # FIXME provide warning and switch to binary2d
            # or not? because it could be multiple categorical... who knows!
            if not np_all(np_sum(data, axis=1) == 1):
                raise ValueError(_TYPE_CHECK_ERROR.format("category"))

        # Match inferred with dist_assumption
        if inferred == "binary" and self.dist_assumption in [
            "binary",
            "classification",
            "auto",
        ]:
            return "binary"
        elif inferred in [
            "multiclass",
            "multilabel-indicator",
        ] and self.dist_assumption in [
            "category",
            "classification",
            "auto",
        ]:
            return "category"
        elif inferred in [
            "continuous",
            "continuous-multioutput",
        ] and self.dist_assumption in [
            "continuous",
            "auto",
        ]:
            return "continuous"
        elif inferred == "multiclass-multioutput":
            raise NotImplementedError("multiclass-multioutput not supported")
            # NOTE we can actually implement this as concatenations of loss
            # functions, but that is up to the future user to decide for now.

        raise ValueError(
            _TYPE_COMPLIANCE_ERROR.format(self.dist_assumption, inferred)
        )

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
            self.inferred_type_ = type_of_target(X)
            self.dist_type_ = self._get_type(X, self.inferred_type_)
        else:
            if not isinstance(X, self.in_type_):
                raise ValueError(
                    _ARG_ERROR_MESSAGE.format("X", "of type " + self.in_type_)
                )
            inferred = type_of_target(X)
            if not inferred == self.inferred_type_:
                raise ValueError(
                    "Inferred distribution type of X does not match "
                    + self.inferred_type_
                )
            if not self._get_type(X, inferred) == self.dist_type_:
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

    def fit(self, X):
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
