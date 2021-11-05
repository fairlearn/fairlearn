# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from pandas import Series
from numpy import number


class FloatTransformer:
    """
    Transformer that maps dataframes to numpy arrays of floats.

    Applies one-hot-encoding to categorical columns, standard scalar to
    numerical columns.
    """

    def __init__(self):
        """Initialize empty transformers."""
        self.x = None
        self.y = None
        self.z = None

    def fit(self, X, y, sensitive_features):
        """Fit the three transformers."""
        X, y, sensitive_features = self._check(X, y, sensitive_features)
        self.x = make_preprocessor(X)
        self.y = make_preprocessor(y)
        self.z = make_preprocessor(sensitive_features)

        self.x.fit(X)
        self.y.fit(y)
        self.z.fit(sensitive_features)

    def transform(self, X, y, sensitive_features):
        """Apply the three transformers."""
        return (self.x.transform(X), self.y.transform(y), self.z.transform(sensitive_features))

    def fit_transform(self, X, y, sensitive_features):
        """Apply fit and return the transformed."""
        X, y, sensitive_features = self._check(X, y, sensitive_features)
        self.fit(X, y, sensitive_features)
        return self.transform(X, y, sensitive_features)

    def _check(self, X, y, z):
        """Check whether the datas are satisfying and return converted input."""
        return _check(X), _check(y), _check(z)


def _check(X):
    """Check whether a single dataframe is okay and convert faults."""
    # TODO
    if isinstance(X, Series):
        X = X.to_frame()
    return X


def make_preprocessor(X):
    """Make the Transformer for a single dataframe."""
    ct = make_column_transformer(
        (StandardScaler(),
         make_column_selector(dtype_include=number)),
        (OneHotEncoder(drop='if_binary', sparse=False),
         make_column_selector(dtype_include="category")))
    return ct
