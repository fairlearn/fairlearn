# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector, \
    ColumnTransformer
from pandas import Series
from numpy import number

class FloatTransformer:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None

    def fit(self, X, y, sensitive_features):
        X, y, sensitive_features = self._check(X, y, sensitive_features)
        self.x = make_preprocessor(X)
        self.y = make_preprocessor(y)
        self.z = make_preprocessor(sensitive_features)

        self.x.fit(X)
        self.y.fit(y)
        self.z.fit(sensitive_features)

    def transform(self, X, y, sensitive_features):
        return (self.x.transform(X), self.y.transform(y), self.z.transform(sensitive_features))
    
    def fit_transform(self, X, y, sensitive_features):
        X, y, sensitive_features = self._check(X, y, sensitive_features)
        self.fit(X, y, sensitive_features)
        return self.transform(X, y, sensitive_features)
    
    def _check(self, X, y, z):
        return _check(X), _check(y), _check(z)

def _check(X):
    # TODO
    if isinstance(X, Series):
        X = X.to_frame()
    return X

def make_preprocessor(X):
    ct = make_column_transformer(
        (StandardScaler(),
         make_column_selector(dtype_include=number)),
        (OneHotEncoder(drop='if_binary', sparse=False),
         make_column_selector(dtype_include="category")))
    return ct