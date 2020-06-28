# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class CorrelationRemover(BaseEstimator, TransformerMixin):
    r"""
    A component that filters out sensitive correlations in a dataset.

    CorrelationRemover applies a linear transformation to the non-sensitive feature columns in order
    to remove their correlation with the sensitive feature columns while retaining as much information
    as possible (as measured by the least-squares error).

    Parameters
    ----------
    sensitive_feature_ids : list of columns to filter out this can be a sequence of either int
      ,in the case of numpy, or string, in the case of pandas.
    alpha : parameter to control how much to filter, for alpha=1.0 we filter out
      all information while for alpha=0.0 we don't apply any.
    center : setting to tell if this preprocessing step should center the data for
      numerical stability

    Notes
    -----

    This method will change the original dataset by removing all correlation with sensitive values.
    To describe that mathematically, let's assume in the original dataset :math:`X` we've got a set of
    sensitive atttributes :math:`S` and a set of non-sensitive attributes :math:`Z`. Mathmatically this method
    will be solving the following problem.

    .. math::

       \min _{\mathbf{z}_{1}, \ldots, \mathbf{z}_{n}} \sum_{i=1}^{n}\left\|\mathbf{z}_{i}-\mathbf{x}_{i}\right\|^{2} \\
       \text{subject to} \\
       \frac{1}{n} \sum_{i=1}^{n} \mathbf{z}_{i}\left(\mathbf{s}_{i}-\overline{\mathbf{s}}\right)^{T}=\mathbf{0}


    The columns in :math:`S` will be dropped but the hyper parameter :math:`\alpha` does allow you to tweak
    the amount of filtering that gets applied.

    .. math::

      X_{\text{tfm}} = \alpha X_{\text{filtered}} + (1-\alpha) X_{\text{orig}}
    """

    def __init__(self, sensitive_feature_ids=None, alpha=1.0, center=True):
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
        if (not self.columns) or (len(self.columns) == 0):
                raise ValueError(f"No sensitive feature ids were passed to this object, got {self.columns}")
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
