# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

import fairlearn.utils._compatibility as compat

# FIXME: memoize type_of_target. It is quite expensive and called repeatedly.


class FloatTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts input data to numpy arrays of floats.

    This class acts as a wrapper around scikit-learn transformers, automatically
    inferring the data type and applying appropriate transformations. It serves
    as a meta-transformer with the following key features:

    1. Automatically detects the input data type (categorical or numerical).
    2. Applies one-hot encoding to categorical data.
    3. Passes numerical data through without modification.
    4. Ensures all output is in the form of floating-point numpy arrays.

    Attributes:
        transformer : str, sklearn.base.TransformerMixin, or None
            Specifies the transformation method. Can be "auto", a specific transformer
            name (e.g., "one_hot_encoder"), None for pass-through, or a custom transformer object.

    Note:
        When using "auto", the class will attempt to choose the most appropriate
        transformation based on the input data type.
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
        if X.ndim == 1:
            X = X.reshape(-1, 1)
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
            if self.inferred_type_ in ["continous-multioutut", "multiclass-multioutput"]:
                raise ValueError("Multioutput not supported")
        return self

    def transform(self, X):
        """Transform X using the fitted encoder or passthrough."""
        if isinstance(self.transformer, str) or self.transformer is None:
            if not type_of_target(X) == self.inferred_type_:
                raise ValueError("Unknown label type")
            return (
                self.transform_.transform(self._check(X)).astype(float)
                if self.inferred_type_ in ["binary", "multiclass"]
                else self._check(X, dtype=float)
            )
        else:
            return self.transform_.transform(X)

    def inverse_transform(self, y):
        """Transform y back to X using the inverse transform of the encoder."""
        if self.transformer is None or isinstance(self.transformer, str):
            if self.inferred_type_ == "continuous":
                inverse = y
            else:
                inverse = self.transform_.inverse_transform(y)

        return inverse.reshape(-1) if self.input_dim_ == 1 else inverse
