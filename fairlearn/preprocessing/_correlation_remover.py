# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


class CorrelationRemover(BaseEstimator, TransformerMixin):
    r"""
    A component that filters out sensitive correlations in a dataset.

    CorrelationRemover applies a linear transformation to the non-sensitive feature columns
    in order to remove their correlation with the sensitive feature columns while retaining
    as much information as possible (as measured by the least-squares error).

    Read more in the :ref:`User Guide <preprocessing>`.

    Parameters
    ----------
        sensitive_feature_ids : list
            list of columns to filter out this can be a sequence of
            either int ,in the case of numpy, or string, in the case of pandas.
        alpha : float
            parameter to control how much to filter, for alpha=1.0 we filter out
            all information while for alpha=0.0 we don't apply any.

    Notes
    -----
    This method will change the original dataset by removing all correlation with sensitive
    values. To describe that mathematically, let's assume in the original dataset :math:`X`
    we've got a set of sensitive attributes :math:`S` and a set of non-sensitive attributes
    :math:`Z`.

    In mathematical terms, assume we have the original dataset :math:`X`, which
    contains a set of **sensitive attributes** denoted by :math:`S` and a set of
    **non-sensitive attributes** denoted by :math:`Z`. The goal is to remove
    correlations between the sensitive attributes and the non-sensitive attributes.

    Let :math:`m_s` and :math:`m_{ns}` denote the number of sensitive and non-sensitive
    features, respectively.
    Let :math:`\bar{S}` represent the mean of the sensitive attributes, *i.e.*,
    :math:`\bar{S} = (\bar{s}_1, \dots, \bar{s}_{m_s})`, where
    :math:`\bar{s}_j` is the mean of the :math:`j\text{-th}` sensitive feature.

    For each non-sensitive feature :math:`z_j` in :math:`Z`,
    we compute an optimal weight vector :math:`w_j^* \in \mathbb{R}^{m_s}` that minimizes the
    following least squares objective:

    .. math::

        \min _{w} \| z_j - (S - \bar{S}) w \|_2^2

    In other words, :math:`w_j^*` is the solution to a linear regression problem where we project
    :math:`z_j` onto the centered sensitive attributes. The weight matrix
    :math:`W^* = (w_1^*, \dots, w_{m_{ns}}^*)` is thus obtained by solving this regression
    for each non-sensitive feature.

    Once we have the optimal weight matrix :math:`W^*`, we compute the **residual
    non-sensitive attributes** :math:`Z^*` as follows:

    .. math::

        Z^* = Z - (S - \bar{S}) W^*


    The columns in :math:`S` will be dropped from the dataset :math:`X`, and :math:`Z^*` will replace
    the original non-sensitive features :math:`Z`, but the hyper parameter :math:`\alpha`
    does allow you to tweak the amount of filtering that gets applied:

    .. math::

        X_{\text{tfm}} = \alpha X_{\text{filtered}} + (1-\alpha) X_{\text{orig}}

    Note that the lack of correlation does not imply anything about statistical dependence.
    Therefore, we expect this to be most appropriate as a preprocessing step for
    (generalized) linear models.

    .. versionadded:: 0.6

    """

    def __init__(self, *, sensitive_feature_ids=None, alpha=1):
        self.sensitive_feature_ids = sensitive_feature_ids
        self.alpha = alpha

    def _split_X(self, X):
        """Split up X into a sensitive and non-sensitive group."""
        sensitive = [self.lookup_[i] for i in self.sensitive_feature_ids]
        non_sensitive = [i for i in range(X.shape[1]) if i not in sensitive]
        return X[:, non_sensitive], X[:, sensitive]

    def _create_lookup(self, X):
        """Create a lookup to handle column names correctly."""
        if isinstance(X, pd.DataFrame):
            self.lookup_ = {c: i for i, c in enumerate(X.columns)}
            return X.values
        # correctly handle a 1d input
        if len(X.shape) == 1:
            return {0: 0}
        self.lookup_ = {i: i for i in range(X.shape[1])}
        return X

    def fit(self, X, y=None):
        """Learn the projection required to make the dataset uncorrelated with sensitive columns."""  # noqa: E501
        self._create_lookup(X)
        X = self._validate_data(X)
        X_use, X_sensitive = self._split_X(X)
        # correctly handle zero provided sensitive features
        if X_sensitive.shape[1] == 0:
            self.sensitive_mean_ = np.array([])
        else:
            self.sensitive_mean_ = X_sensitive.mean()
        X_s_center = X_sensitive - self.sensitive_mean_
        self.beta_, _, _, _ = np.linalg.lstsq(X_s_center, X_use, rcond=None)
        self.X_shape_ = X.shape
        return self

    def transform(self, X):
        """Transform X by applying the correlation remover."""
        X = check_array(X, estimator=self)
        check_is_fitted(self, ["beta_", "X_shape_", "lookup_", "sensitive_mean_"])
        if self.X_shape_[1] != X.shape[1]:
            raise ValueError(
                f"The trained data has {self.X_shape_[1]} features while this dataset"
                f" has {X.shape[1]}."
            )
        X_use, X_sensitive = self._split_X(X)
        X_s_center = X_sensitive - self.sensitive_mean_
        X_filtered = X_use - X_s_center.dot(self.beta_)
        X_use = np.atleast_2d(X_use)
        X_filtered = np.atleast_2d(X_filtered)
        return self.alpha * X_filtered + (1 - self.alpha) * X_use

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_transformer_data_not_an_array": (
                    "this estimator only accepts pandas dataframes or numpy ndarray as input."
                ),
            }
        }
