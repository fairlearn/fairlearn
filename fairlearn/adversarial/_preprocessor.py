# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from numpy import unique
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

import fairlearn.utils._compatibility as compat

# FIXME: memoize type_of_target. It is quite expensive and called repeatedly.


class FloatTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that maps dataframes to numpy arrays of floats.

    It is in essence a wrapper around sklearn transformers (meta-transformer)
    that automatically infers the type of data, or makes sure the transformer
    encodes it as floating point numbers. It applies one-hot-encoding
    to categorical columns, and 'passthrough' (nothing) to
    numerical columns. The usefulness of this class is that it can
    preprocess different kinds of data (discrete, continuous) to one
    standard format while also remembering what kind of data was inputted.
    """

    def __init__(self, transformer="auto"):
        """
        Initialize empty transformers with the given distribution assumption.

        Parameters
        ----------
        transformer : str, sklearn.base.TransformerMixin, optional, default = "auto"
            This is a string that indicates the transformer, such as
            :code:`"auto"`, :code:`"one_hot_encoder"`, :code:`"binarizer"`.
            Or, None, for pass-through. Or, a transformer object.
        """
        self.transformer = transformer

    def _check(self, X, dtype=None, init=False):
        """
        Check X and convert to 2d ndarray.

        dtype : numpy.dtype
            None to keep dtypes, float to coerce to numeric.

        init : bool
            Whether this is the first call to _check or not. Useful to store
            the dimensions of X, so we can use this for inverse_transform

        Returns
        -------
        X : numpy.ndarray
            validated input
        """
        X = check_array(
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=dtype,
            ensure_2d=False,
        )
        if init:
            self.input_dim_ = X.ndim
            if self.input_dim_ > 2:
                raise ValueError("Data can be at most two dimensional, not %d" % self.input_dim_)
        else:
            if X.ndim != self.input_dim_:
                raise ValueError("Dimension of data is inconsistent with previous call")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise SystemError("Data must be two dimensional at this point.")

        return X

    def fit(self, X, y=None):
        """Fit the three transformers."""
        # Sci-kit learn parameter
        if isinstance(self.transformer, str) or self.transformer is None:
            self.inferred_type_ = type_of_target(X)
            X = self._check(X, init=True)
            self.n_features_in_, self.n_features_out_ = X.shape[0], X.shape[1]

            if self.inferred_type_ in ["binary", "multiclass"]:
                # NOTE: if 'binary' then it could be possible it is already 0/1
                # encoded. So may want to skip redundant OHE in that case...
                self.transform_ = OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="error",
                    **compat._SPARSE_OUTPUT_FALSE,
                )
                self.transform_.fit(X)
                self.n_features_out_ = sum(
                    len(cat) if len(cat) != 2 else 1 for cat in self.transform_.categories_
                )
            # elif "multilabel-indicator" needn't be encoded, so we do not create
            # an encoder then.
        else:
            # It is useful to gather n_features_out_ to use in constructing NN
            # After, we discard the checked data and just feed unchecked data
            # to the transformer.
            X_temp = self._check(X, init=True)
            self.n_features_in_ = X_temp.shape[0]
            if type_of_target(X_temp) in ["multiclass", "multilabel-indicator"]:
                self.n_features_out_ = unique(X_temp)
            else:
                self.n_features_out_ = self.n_features_in_

            self.transform_ = self.transformer
            self.transform_.fit(X)
            if hasattr(self.transform_, "n_features_out_"):
                self.n_features_out_ = self.transform_.n_features_out_

        return self

    def transform(self, X):
        """Transform X using the fitted encoder or passthrough."""
        if isinstance(self.transformer, str) or self.transformer is None:
            if not type_of_target(X) == self.inferred_type_:
                raise ValueError(
                    "Inferred distribution type of X does not match " + self.inferred_type_
                )
            return (
                self.transform_.transform(self._check(X)).astype(float)
                if self.inferred_type_ in ["binary", "multiclass"]
                else self._check(X, dtype=float)
            )
        else:
            return self.transform_.transform(X)

    def inverse_transform(self, y):
        """Transform y back to X using the inverse transform of the encoder."""
        if (
            self.transformer is None
            or isinstance(self.transformer, str)
            and self.inferred_type_
            in ["continuous", "continuous-multioutput", "multilabel-indicator"]
        ):
            inverse = y
        else:
            inverse = self.transform_.inverse_transform(y)

        return inverse.reshape(-1) if self.input_dim_ == 1 else inverse
<<<<<<< HEAD
=======


def _get_type(data, assumption):
    """Get the type (binary, category, continuous) of the data under assump."""
    inferred = type_of_target(data)
    if inferred == "multilabel-indicator":
        #  Design choice: multiple binary columns are not supported.
        #                 multiple columns may only be one-hot encoding
        # FIXME provide warning and switch to binary2d
        # or not? because it could be multiple categorical... who knows!
        if not np_all(np_sum(data, axis=1) == 1):
            raise ValueError(_TYPE_CHECK_ERROR.format("category"))
        u = unique(data)
        if len(u) != 2 or (not isin([0, 1], u).all()):
            raise ValueError(_INVALID_OHE)

    # Match inferred with dist_assumption
    if inferred == "binary" and assumption in [
        "binary",
        "classification",
        "continuous",
        "auto",
    ]:
        return "binary"
    elif inferred in ["multiclass", "multilabel-indicator"] and assumption in [
        "category",
        "continuous",
        "classification",
        "auto",
    ]:
        return "category"
    elif inferred in ["continuous", "continuous-multioutput"] and assumption in [
        "continuous",
        "binary",
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
>>>>>>> 9a97c4d (Fix several failing estimator checks)
