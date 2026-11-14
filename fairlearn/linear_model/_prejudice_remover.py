# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""Logistic regression with the prejudice remover regularizer."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted, validate_data

from fairlearn.utils._input_validation import _check_no_missing_values

_KW_SENSITIVE_FEATURES = "sensitive_features"

# Probability values are clipped away from 0 and 1 before taking logarithms,
# so that a saturated fit reports a very small prejudice index instead of -inf.
_PROBABILITY_EPSILON = 1e-10


class PrejudiceRemover(ClassifierMixin, BaseEstimator):
    r"""Binary classifier that regularizes away the influence of a sensitive feature.

    This is the logistic regression model of *Fairness-Aware Classifier with
    Prejudice Remover Regularizer* by Kamishima et al. :footcite:`kamishima2012prejudice`.
    It is an in-processing technique aimed at demographic parity: instead of
    post-processing the predictions of a fitted model, the training objective
    itself includes a term that penalizes the statistical dependence between the
    sensitive feature and the predictions.

    Read more in the :ref:`User Guide <prejudice_remover>`.

    .. versionadded:: 0.16.0

    Parameters
    ----------
    eta : float, default=1.0
        Weight of the prejudice remover regularizer. With :code:`eta=0` no
        fairness penalty is applied and the estimator reduces to one logistic
        regression per group. Larger values trade predictive accuracy for
        predictions that depend less on the sensitive feature.

    C : float, default=1.0
        Inverse of the strength of the L2 regularization: the loss includes a
        term :math:`\lVert \mathbf{W} \rVert_2^2 / (2C)`, matching the meaning
        of :code:`C` in :class:`sklearn.linear_model.LogisticRegression`.
        Must be strictly positive. With :code:`eta=0`, the fit equals the fit of
        a :class:`sklearn.linear_model.LogisticRegression` with the same
        :code:`C`, fitted separately on each group.

    fit_intercept : bool, default=True
        Whether an intercept should be learned for each group.

    max_iter : int, default=100
        Maximum number of iterations of the L-BFGS-B optimizer.

    tol : float, default=1e-4
        Tolerance for the gradient norm, as used by the optimizer as its
        stopping criterion.

    Attributes
    ----------
    classes_ : numpy.ndarray
        The two class labels, sorted.

    coef_ : numpy.ndarray of shape (n_groups_, n_features)
        The coefficients of each group, ordered as in
        :attr:`sensitive_feature_values_`.

    intercept_ : numpy.ndarray of shape (n_groups_,)
        The intercept of each group. It is :code:`0.0` for every group when
        :code:`fit_intercept=False`.

    sensitive_feature_values_ : numpy.ndarray of shape (n_groups_,)
        The distinct values the sensitive feature takes in the training data,
        sorted. A group is one such value.

    prejudice_index_ : float
        The value of the prejudice remover regularizer at the solution, as
        defined by equation (11) of :footcite:`kamishima2012prejudice`. It is a
        sum over the training samples, and it is :code:`0.0` when the
        predictions are independent of the sensitive feature.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : numpy.ndarray of shape (n_features_in_,)
        Names of features seen during fit. Only defined when :code:`X` has
        string feature names.

    n_iter_ : int
        Number of iterations run by the optimizer.

    Notes
    -----
    Let :math:`S` be the sensitive feature and let :math:`\mathbf{x}` be the
    remaining features. The model fits one weight vector
    :math:`\mathbf{w}_s` per value :math:`s` of :math:`S`, so that

    .. math::

        M[y \mid \mathbf{x}, s; \Theta]
        = y \sigma(\mathbf{x}^\top \mathbf{w}_s)
          + (1 - y) (1 - \sigma(\mathbf{x}^\top \mathbf{w}_s))

    where :math:`\sigma` is the logistic function and
    :math:`\Theta = \{\mathbf{w}_s\}_s`. The objective is

    .. math::

        - \sum_{(y_i, \mathbf{x}_i, s_i) \in D} \ln M[y_i \mid \mathbf{x}_i, s_i; \Theta]
        + \eta R_{PR}(D, \Theta)
        + \frac{1}{2C} \sum_s \lVert \mathbf{w}_s \rVert_2^2,

    where the prejudice remover regularizer is

    .. math::

        R_{PR}(D, \Theta) = \sum_{(\mathbf{x}_i, s_i) \in D} \sum_{y \in \{0, 1\}}
        M[y \mid \mathbf{x}_i, s_i; \Theta]
        \ln \frac{\Pr[\hat{Y} = y \mid s_i]}{\Pr[\hat{Y} = y]}.

    :math:`R_{PR}` is the prejudice index, a sample estimate of the mutual
    information between the predictions :math:`\hat{Y}` and the sensitive
    feature :math:`S`. It grows when the predictions are mostly explained by
    :math:`S`, so minimizing it makes :math:`S` less influential. It is
    estimated on the training set, so a large :code:`eta` makes the
    *training* predictions independent of :math:`S`; the effect on unseen data
    depends on the data and on :code:`eta`.

    Only one categorical sensitive feature with a finite set of values is
    supported. Continuous sensitive features and several sensitive features are
    not, because the regularizer needs a group for each value.

    Because the model keeps one weight vector per group, the sensitive feature
    has to be passed to :meth:`predict` and :meth:`predict_proba` as well as to
    :meth:`fit`.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    >>> import numpy as np
    >>> from fairlearn.linear_model import PrejudiceRemover
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> sensitive = rng.integers(0, 2, size=n)
    >>> X = rng.normal(size=(n, 2))
    >>> y = (X[:, 0] + 2 * sensitive + rng.normal(size=n) > 1).astype(int)
    >>> remover = PrejudiceRemover(eta=10.0).fit(X, y, sensitive_features=sensitive)
    >>> remover.predict(X, sensitive_features=sensitive).shape
    (200,)
    """

    def __init__(
        self,
        *,
        eta: float = 1.0,
        C: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        self.eta = eta
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y, *, sensitive_features):
        """Fit the model.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The non-sensitive feature matrix.

        y : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            The labels. Exactly two distinct labels are supported.

        sensitive_features : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            A single categorical feature identifying the group of each sample.

        Returns
        -------
        self : PrejudiceRemover
            The fitted estimator.

        Raises
        ------
        ValueError
            If the labels are not binary, if the sensitive feature contains
            missing values or more than one column, or if :code:`eta` or
            :code:`C` is out of range.
        """
        if self.eta < 0:
            raise ValueError(f"eta must be non-negative, but got {self.eta}.")
        if self.C <= 0:
            raise ValueError(f"C must be strictly positive, but got {self.C}.")

        X, y = validate_data(self, X, y, accept_sparse=False, dtype=np.float64)

        groups = self._validate_sensitive_features(sensitive_features, X.shape[0])

        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                f"PrejudiceRemover supports binary classification, but y has "
                f"{self.classes_.size} distinct values."
            )
        y_binary = (y == self.classes_[1]).astype(np.float64)

        self.sensitive_feature_values_ = np.unique(groups)
        group_index = self._group_index(groups)

        design = np.hstack([X, np.ones((X.shape[0], 1))]) if self.fit_intercept else X
        n_groups = self.sensitive_feature_values_.size

        result = minimize(
            self._objective,
            np.zeros(n_groups * design.shape[1]),
            args=(design, y_binary, group_index),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        if not result.success:
            warnings.warn(
                f"PrejudiceRemover did not converge within {self.max_iter} iterations: "
                f"{result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )

        weights = result.x.reshape(n_groups, design.shape[1])
        if self.fit_intercept:
            self.coef_ = weights[:, :-1].copy()
            self.intercept_ = weights[:, -1].copy()
        else:
            self.coef_ = weights.copy()
            self.intercept_ = np.zeros(n_groups)
        self.n_iter_ = int(result.nit)

        probabilities = expit(np.einsum("ij,ij->i", design, weights[group_index]))
        terms, _ = self._prejudice_index(probabilities, group_index)
        self.prejudice_index_ = float(terms.sum())
        return self

    def predict(self, X, *, sensitive_features):
        """Predict the label of each sample.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The non-sensitive feature matrix.

        sensitive_features : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            The sensitive feature of each sample. Every value must have been
            seen during fit.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The predicted labels.
        """
        probabilities = self.predict_proba(X, sensitive_features=sensitive_features)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, X, *, sensitive_features):
        """Predict the probability of each class.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            The non-sensitive feature matrix.

        sensitive_features : numpy.ndarray, pandas.DataFrame, pandas.Series, or list
            The sensitive feature of each sample. Every value must have been
            seen during fit.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            The probability of each class, ordered as in :attr:`classes_`.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False, accept_sparse=False, dtype=np.float64)
        groups = self._validate_sensitive_features(sensitive_features, X.shape[0])

        unknown = np.setdiff1d(np.unique(groups), self.sensitive_feature_values_)
        if unknown.size > 0:
            raise ValueError(
                f"{_KW_SENSITIVE_FEATURES} contains values that were not seen during fit: "
                f"{unknown.tolist()}."
            )
        group_index = self._group_index(groups)

        design = np.hstack([X, np.ones((X.shape[0], 1))]) if self.fit_intercept else X
        weights = (
            np.column_stack([self.coef_, self.intercept_]) if self.fit_intercept else self.coef_
        )
        probabilities = expit(np.einsum("ij,ij->i", design, weights[group_index]))
        return np.column_stack([1 - probabilities, probabilities])

    def _group_index(self, groups: np.ndarray) -> np.ndarray:
        """Map the values of the sensitive feature to positions in sensitive_feature_values_."""
        return np.searchsorted(self.sensitive_feature_values_, groups)

    def _validate_sensitive_features(self, sensitive_features, n_samples: int) -> np.ndarray:
        """Flatten the sensitive feature and check that it matches the samples."""
        groups = np.asarray(
            sensitive_features.to_numpy()
            if hasattr(sensitive_features, "to_numpy")
            else sensitive_features
        )
        if groups.ndim == 2 and groups.shape[1] == 1:
            groups = groups[:, 0]
        if groups.ndim != 1:
            raise ValueError(
                f"{_KW_SENSITIVE_FEATURES} must be a single feature, but it has shape "
                f"{groups.shape}."
            )
        if groups.shape[0] != n_samples:
            raise ValueError(
                f"{_KW_SENSITIVE_FEATURES} has {groups.shape[0]} values, but X has "
                f"{n_samples} samples."
            )
        _check_no_missing_values(groups, _KW_SENSITIVE_FEATURES)
        return groups

    def _prejudice_index(
        self, probabilities: np.ndarray, group_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Return the prejudice index per sample and the per-sample gradient factor.

        The regularizer of the paper sums, over samples :math:`i` and classes
        :math:`y`, the term

        .. math::

            M[y \mid \mathbf{x}_i, s_i] \ln
            \frac{\Pr[\hat{Y} = y \mid s_i]}{\Pr[\hat{Y} = y]},

        where :math:`\Pr[\hat{Y} \mid s]` and :math:`\Pr[\hat{Y}]` are estimated
        from the same predictions (equations (9) and (10) of the paper). The
        per-sample terms are returned alongside the factor

        .. math::

            a_i = \ln \frac{\rho_{s_i}}{1 - \rho_{s_i}} - \ln \frac{\pi}{1 - \pi},

        where :math:`\rho_s = \Pr[\hat{Y} = 1 \mid s]` and
        :math:`\pi = \Pr[\hat{Y} = 1]`, because the derivative of the regularizer
        with respect to the weight vector of group :math:`g` reduces to
        :math:`\sum_{i \in g} a_i \rho_i (1 - \rho_i) \mathbf{x}_i`: the
        derivatives through :math:`\rho_s` and :math:`\pi` cancel, as
        :math:`\sum_{i \in s} \hat{p}_i = |D_s| \rho_s` and
        :math:`\sum_i \hat{p}_i = |D| \pi` hold by definition.
        """
        group_sizes = np.bincount(group_index)
        group_means = (
            np.bincount(group_index, weights=probabilities, minlength=group_sizes.size)
            / group_sizes
        )
        overall_mean = probabilities.mean()

        # A saturated fit drives a group mean to 0 or 1, where the logarithm is
        # unbounded. Clipping keeps the objective finite and, because both the
        # value and the factor use the clipped means, leaves the gradient exact
        # for the clipped objective.
        clipped_group_means = np.clip(group_means, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
        clipped_overall_mean = float(
            np.clip(overall_mean, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
        )

        means_per_sample = clipped_group_means[group_index]
        terms = probabilities * (np.log(means_per_sample) - np.log(clipped_overall_mean)) + (
            1 - probabilities
        ) * (np.log1p(-means_per_sample) - np.log1p(-clipped_overall_mean))
        factor = (
            np.log(means_per_sample)
            - np.log1p(-means_per_sample)
            - (np.log(clipped_overall_mean) - np.log1p(-clipped_overall_mean))
        )
        return terms, factor

    def _objective(
        self, weights_flat: np.ndarray, design: np.ndarray, y: np.ndarray, group_index: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return the objective of the paper and its gradient with respect to the weights."""
        n_groups = int(group_index.max()) + 1
        weights = weights_flat.reshape(n_groups, design.shape[1])
        linear = np.einsum("ij,ij->i", design, weights[group_index])
        probabilities = expit(linear)

        clipped = np.clip(probabilities, _PROBABILITY_EPSILON, 1 - _PROBABILITY_EPSILON)
        negative_log_likelihood = -np.sum(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))

        terms, factor = self._prejudice_index(probabilities, group_index)
        prejudice_index = terms.sum()
        # As in scikit-learn, the intercept is not penalised.
        penalised = weights[:, :-1] if self.fit_intercept else weights
        l2 = np.sum(penalised * penalised) / (2 * self.C)

        value = negative_log_likelihood + self.eta * prejudice_index + l2

        # d(value)/d(linear) for the loss and for the regularizer, chained
        # through the logistic derivative.
        per_sample = (probabilities - y) + self.eta * factor * probabilities * (1 - probabilities)
        weighted = per_sample[:, np.newaxis] * design
        gradient = np.empty_like(weights)
        for group in range(n_groups):
            gradient[group] = weighted[group_index == group].sum(axis=0)
        gradient[:, : penalised.shape[1]] += penalised / self.C
        return float(value), gradient.ravel()
