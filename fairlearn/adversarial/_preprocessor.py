# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from typing import Literal, Self

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted


class FloatTransformer(TransformerMixin, BaseEstimator):
    """Transformer that converts input data to numpy arrays of floats.

    This class acts as a wrapper around scikit-learn transformers, automatically
    inferring the data type and applying appropriate transformations. It serves
    as a meta-transformer with the following key features:

    1. Automatically detects the input data type (categorical or numerical).
    2. Applies one-hot encoding to categorical data.
    3. Passes numerical data through without modification.
    4. Ensures all output is in the form of floating-point numpy arrays.

    Attributes
    ----------
    transformer : str, sklearn.base.TransformerMixin, or None
        Specifies the transformation method. Can be "auto", None for pass-through, or a
        custom transformer object.

    Notes
    -----
    When using "auto", the class will attempt to choose the most appropriate
    transformation based on the input data type.
    """

    def __init__(self, transformer: Literal["auto"] | TransformerMixin | None = "auto"):
        """Initialize empty transformers with the given distribution assumption.

        Parameters
        ----------
        transformer : str, sklearn.base.TransformerMixin, optional, default = "auto"
            If a string is passed it has to be :code:`"auto"`.
            Use :code:`None` for pass-through.
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

    def fit(self, X, y=None) -> Self:
        self.inferred_type_ = type_of_target(X)
        X = self._check(X, init=True)
        self.n_features_in_, self.n_features_out_ = X.shape[0], X.shape[1]

        passthrough_transformer = FunctionTransformer(func=None)
        if self.transformer is None:
            self.transform_ = passthrough_transformer
            self.transform_.fit(X)
            return self

        if self.transformer == "auto":
            if self.inferred_type_ in ["binary", "multiclass"]:
                # NOTE: if 'binary' then it could be possible it is already 0/1
                # encoded. So may want to skip redundant OHE in that case...
                self.transform_ = OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="error",
                    sparse_output=False,
                )
                self.transform_.fit(X)
                self.n_features_out_ = sum(
                    len(cat) if len(cat) != 2 else 1 for cat in self.transform_.categories_
                )
            else:
                self.transform_ = passthrough_transformer
                self.transform_.fit(X)
            return self

        self.transform_ = self.transformer
        self.transform_.fit(X)
        return self

    def transform(self, X):
        """Transform X using the fitted encoder or passthrough."""
        check_is_fitted(self)

        if self.transformer == "auto" or self.transformer is None:
            return (
                self.transform_.transform(self._check(X)).astype(float)
                if self.inferred_type_ in ["binary", "multiclass"]
                else self._check(X, dtype=float)
            )

        return self.transform_.transform(self._check(X, dtype=float))

    def inverse_transform(self, y):
        """Transform y back to X using the inverse transform of the encoder."""
        check_is_fitted(self)
        inverse = self.transform_.inverse_transform(self._check(y, dtype=float))
        return inverse.reshape(-1) if self.input_dim_ == 1 else inverse
