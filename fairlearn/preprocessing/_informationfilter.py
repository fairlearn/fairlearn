import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler


def _scalar_projection(vec, unto):
    return vec.dot(unto) / unto.dot(unto)


def _vector_projection(vec, unto):
    return _scalar_projection(vec, unto) * unto


class InformationFilter(BaseEstimator, TransformerMixin):
    r"""
    The `InformationFilter` filters out sensitive correlations in a dataset.

    The `InformationFilter` uses a variant of the gram smidt process
    to filter information out of the dataset. This can be useful if you
    want to filter information out of a dataset because of fairness.
    To explain how it works: given a training matrix :math:`X` that contains
    columns :math:`x_1, ..., x_k`. If we assume columns :math:`x_1` and :math:`x_2`
    to be the sensitive columns then the information-filter will
    remove information by applying these transformations;

    .. math::

       \\begin{split}
       v_1 & = x_1 \\\\
       v_2 & = x_2 - \\frac{x_2 v_1}{v_1 v_1}\\\\
       v_3 & = x_3 - \\frac{x_k v_1}{v_1 v_1} - \\frac{x_2 v_2}{v_2 v_2}\\\\
       ... \\\\
       v_k & = x_k - \\frac{x_k v_1}{v_1 v_1} - \\frac{x_2 v_2}{v_2 v_2}
       \\end{split}

    Concatenating our vectors (but removing the sensitive ones) gives us
    a new training matrix :math:`X_{fair} =  [v_3, ..., v_k]`.

    :param columns: list of columns to filter out this can be a sequence of either int
                    (in the case of numpy) or string (in the case of pandas).
    :param alpha: parameter to control how much to filter, for alpha=1 we filter out
                  all information while for alpha=0 we don't apply any.
    :param center: setting to tell if this preprocessing step should center the data for
                   numerical stability
    """

    def __init__(self, columns, alpha=1, center=True):
        self.columns = columns
        self.alpha = alpha
        self.center = center

    def _check_coltype(self, X):
        check_array(X, estimator=self)
        for col in self.columns:
            if isinstance(col, str):
                if isinstance(X, np.ndarray):
                    raise ValueError(
                        f"column {col} is a string but datatype receive is numpy."
                    )
                if isinstance(X, pd.DataFrame):
                    if col not in X.columns:
                        raise ValueError(f"column {col} is not in {X.columns}")
            if isinstance(col, int):
                if col not in range(np.atleast_2d(np.array(X)).shape[1]):
                    raise ValueError(
                        f"column {col} is out of bounds for input shape {X.shape}"
                    )

    def _col_idx(self, X, name):
        if isinstance(name, str):
            if isinstance(X, np.ndarray):
                raise ValueError(
                    "You cannot have a column of type string on a numpy input matrix."
                )
            return {name: i for i, name in enumerate(X.columns)}[name]
        return name

    def _make_v_vectors(self, X, col_ids):
        vs = np.zeros((X.shape[0], len(col_ids)))
        for i, c in enumerate(col_ids):
            vs[:, i] = X[:, col_ids[i]]
            for j in range(0, i):
                vs[:, i] = vs[:, i] - _vector_projection(vs[:, i], vs[:, j])
        return vs

    def fit(self, X, y=None):
        """Learn the projection required to make the dataset orthogonal to sensitive columns."""
        self._check_coltype(X)
        self.col_ids_ = [v if isinstance(v, int) else self._col_idx(X, v) for v in self.columns]
        X = check_array(X, estimator=self)
        if self.center:
            X = StandardScaler(with_std=False).fit_transform(X)
        X_fair = X.copy()
        v_vectors = self._make_v_vectors(X, self.col_ids_)
        # gram smidt process but only on sensitive attributes
        for i, col in enumerate(X_fair.T):
            for v in v_vectors.T:
                X_fair[:, i] = X_fair[:, i] - _vector_projection(X_fair[:, i], v)
        # we want to learn matrix P: X P = X_fair
        # this means we first need to create X_fair in order to learn P
        self.projection_, resid, rank, s = np.linalg.lstsq(X, X_fair, rcond=None)
        return self

    def transform(self, X):
        """Transforms X by applying the information filter."""
        check_is_fitted(self, ["projection_", "col_ids_"])
        self._check_coltype(X)
        X = check_array(X, estimator=self)
        if self.center:
            X = StandardScaler(with_std=False).fit_transform(X)
        # apply the projection and remove the column we won't need
        X_fair = X @ self.projection_
        X_removed = np.delete(X_fair, self.col_ids_, axis=1)
        X_orig = np.delete(X, self.col_ids_, axis=1)
        return self.alpha * np.atleast_2d(X_removed) + (1 - self.alpha) * np.atleast_2d(
            X_orig
        )
