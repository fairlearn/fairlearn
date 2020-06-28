# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class CorrelationRemover(BaseEstimator, TransformerMixin):
    r"""
    A component that filters out sensitive correlations in a dataset.

    The `InformationFilter` uses a variant of the Gram-Schmidt process
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
    a new training matrix :math:`X_{filtered} =  [v_3, ..., v_k]`. The final output
    is an interpolation between the original dataset (without sensitive columns)
    and the fitlered dataset.

    .. math::

      X_{\text{tfm}} = \alpha X_{\text{filtered}} + (1-\alpha) X_{\text{orig}}

    :param sensitive_feature_ids: list of columns to filter out this can be a sequence of either int
                    (in the case of numpy) or string (in the case of pandas).
    :param alpha: parameter to control how much to filter, for alpha=1 we filter out
                  all information while for alpha=0 we don't apply any.
    :param center: setting to tell if this preprocessing step should center the data for
                   numerical stability
    """

    def __init__(self, sensitive_feature_ids, alpha=1, center=True):
        self.columns = sensitive_feature_ids
        self.alpha = alpha
        self.center = center

    def _split_X(self, X):
        """Split up X into a sensitive and non-sensitive group."""
        sensitive = self.columns
        non_sensitive = [i for i in range(X.shape[1]) if i not in sensitive]
        return X[:, non_sensitive], X[:, sensitive]

    def fit(self, X, y=None):
        """Learn the projection required to make the dataset orthogonal to sensitive columns."""
        X = check_array(X, estimator=self)
        X_use, X_sensitive = self._split_X(X)
        self.sensitive_mean_ = X_sensitive.mean()
        X_s_center = X_sensitive - self.sensitive_mean_
        self.beta_, _, _, _ = np.linalg.lstsq(X_s_center, X_use, rcond=None)
        self.X_shape_ = X.shape
        return self

    def transform(self, X):
        """Transform X by applying the information filter."""
        X = check_array(X, estimator=self)
        check_is_fitted(self, ["beta_", "X_shape_"])
        if self.X_shape_[1] != X.shape[1]:
            raise ValueError(f"The trained data has {self.X_shape_[1]} while this dataset has {X.shape[1]}.")
        X_use, X_sensitive = self._split_X(X)
        X_s_center = X_sensitive - self.sensitive_mean_
        X_filtered = X_use - X_s_center.dot(self.beta_)
        X_use = np.atleast_2d(X_use)
        X_filtered = np.atleast_2d(X_filtered)
        return self.alpha * X_filtered + (1 - self.alpha) * X_use
