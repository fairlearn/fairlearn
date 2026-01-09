from collections.abc import Iterable

import narwhals.stable.v1 as nw
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.validation import check_is_fitted, validate_data


class KamiranCaldersReweighing(TransformerMixin, BaseEstimator):
    r"""
    A reweighing transformer for mitigating bias with respect to sensitive features.

    ``KamiranCaldersReweighing`` assigns instance-level weights such that the
    distribution of the target variable becomes independent of one or more
    sensitive features. This implements the reweighing method proposed by
    Kamiran and Calders (2012).

    The transformer does **not** modify feature values. Instead, it augments the
    input data with a per-sample weight column, which can be passed to downstream
    estimators via a ``sample_weight`` argument during model training.

    The transformer supports NumPy arrays, pandas DataFrames, and Narwhals
    DataFrames as input. When DataFrame-like inputs are used, feature names
    are preserved and validated in a scikit-learn–compatible manner.

    Read more in the :ref:`User Guide <preprocessing>`.

    Parameters
    ----------
    drop_target : bool, default=True
        Whether to drop the target column in the transformed output.
        Note that the current implementation does not modify the target column
        and always returns features augmented with a ``"weight"`` column.

    Notes
    -----
    Let :math:`S` denote a set of sensitive features and :math:`Y` the target
    variable. The goal of reweighing is to enforce statistical independence
    between :math:`S` and :math:`Y` in the weighted dataset.

    For each combination :math:`(s, y)` of sensitive feature values and target
    labels, the reweighing method computes a weight

    .. math::

        w(s, y) = \frac{P(S = s)\,P(Y = y)}{P(S = s, Y = y)}

    where probabilities are estimated empirically from the training data.

    Each instance with sensitive attributes :math:`s` and label :math:`y`
    is assigned the corresponding weight :math:`w(s, y)`. When training a
    model using these weights, the resulting weighted empirical distribution
    satisfies

    .. math::

        P_w(S = s, Y = y) = P(S = s)\,P(Y = y),

    thereby removing dependence between the sensitive features and the target.

    This method is most naturally applied to learning algorithms that support
    sample weighting.

    .. versionadded:: 0.6
    """

    def __init__(self, *, drop_target: bool = True):
        """
        Initialize the reweighing transformer.

        Parameters
        ----------
        drop_target : bool, default=True
            Whether to drop the target column in the transformed output.
            This parameter is currently retained for API compatibility.
        """
        self.drop_target = drop_target

    def fit(self, X, y, *, sensitive_features=None, **fit_params):
        """
        Fit the reweighing transformer on the data.

        During fitting, empirical probabilities involving the sensitive
        features and the target are estimated and stored internally. These
        quantities are later used to compute instance-level weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features. Supported inputs include NumPy arrays,
            pandas DataFrames, and Narwhals DataFrames.

        y : array-like of shape (n_samples,)
            Target labels corresponding to ``X``.

        sensitive_features : iterable of int or str, optional
            Indices or names of columns in ``X`` corresponding to sensitive
            features. If ``None`` or empty, no reweighing is learned and all
            samples receive unit weight during transformation.

        **fit_params : dict
            Additional fit parameters. These are accepted for API
            compatibility and are ignored.

        Returns
        -------
        self : KamiranCaldersReweighing
            Fitted transformer with learned reweighing factors.
        """
        self._is_pandas_ = isinstance(X, pd.DataFrame)
        self._is_nw_ = isinstance(X, nw.DataFrame)

        # ✅ sklearn-compliant validation (sets n_features_in_)
        X_valid, y_valid = validate_data(self, X, y, ensure_2d=True, dtype=None)

        # Store feature names
        if self._is_pandas_:
            self.feature_names_in_ = X.columns.to_list()
        elif self._is_nw_:
            self.feature_names_in_ = X.columns
        else:
            self.feature_names_in_ = None

        n_samples = X_valid.shape[0]

        # Encode categorical features
        if X_valid.dtype.kind in {"O", "U", "S"}:
            self._x_encoder_ = OrdinalEncoder()
            X_enc = self._x_encoder_.fit_transform(X_valid)
        else:
            self._x_encoder_ = None
            X_enc = X_valid

        # Encode target
        if y_valid.dtype.kind in {"O", "U", "S"}:
            self._y_encoder_ = OrdinalEncoder()
            y_enc = self._y_encoder_.fit_transform(y_valid.reshape(-1, 1)).ravel()
        else:
            self._y_encoder_ = None
            y_enc = y_valid

        self._y_array_ = y_enc

        # Handle sensitive features
        if not sensitive_features:
            self.sensitive_idx_ = []
            self._weights_mapping_ = None
            return self

        if isinstance(sensitive_features, Iterable):
            sensitive_features = list(sensitive_features)

        def get_index(f):
            if isinstance(f, int):
                return f
            if self.feature_names_in_ is not None:
                return self.feature_names_in_.index(f)
            return f

        self.sensitive_idx_ = [get_index(f) for f in sensitive_features]

        S = X_enc[:, self.sensitive_idx_]
        SY = np.column_stack([S, y_enc])

        unique_rows, counts = np.unique(SY, axis=0, return_counts=True)
        ps = np.array([(S == row[:-1]).all(axis=1).mean() for row in unique_rows])
        py = np.array([(y_enc == row[-1]).mean() for row in unique_rows])
        psy = counts / n_samples

        self._weights_mapping_ = {
            tuple(row): (p_s * p_y) / p_sy for row, p_s, p_y, p_sy in zip(unique_rows, ps, py, psy)
        }

        return self

    def transform(self, X, y=None):
        """
        Transform the input data by appending instance-level weights.

        If sensitive features were specified during fitting and ``y`` is
        provided, each sample is assigned a weight according to the
        Kamiran–Calders reweighing scheme. Otherwise, all samples receive
        unit weight.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to transform. Must have the same number and order
            of features as the data used during fitting.

        y : array-like of shape (n_samples,), optional
            Target labels corresponding to ``X``. Required to compute
            non-unit weights when sensitive features were specified at fit
            time.

        Returns
        -------
        X_transformed : array-like or DataFrame
            The input data augmented with an additional ``"weight"`` column.
            The return type matches the input type: NumPy array for array-like
            input, pandas DataFrame for pandas input, and Narwhals DataFrame
            for Narwhals input.
        """
        check_is_fitted(self)

        # ✅ sklearn-compliant feature consistency check
        X_valid = validate_data(self, X, reset=False, ensure_2d=True, dtype=None)

        X_enc = self._x_encoder_.transform(X_valid) if self._x_encoder_ else X_valid
        n_samples = X_valid.shape[0]

        if self._weights_mapping_ is None or y is None:
            weights = np.ones(n_samples)
        else:
            y_array = np.asarray(y).ravel()
            if self._y_encoder_:
                y_array = self._y_encoder_.transform(y_array.reshape(-1, 1)).ravel()

            S = X_enc[:, self.sensitive_idx_]
            structured = np.column_stack([S, y_array])
            weights = np.array([self._weights_mapping_.get(tuple(row), 1.0) for row in structured])

        if self._is_pandas_:
            df_out = pd.DataFrame(X_valid, columns=self.feature_names_in_)
            df_out["weight"] = weights
            return df_out
        elif self._is_nw_:
            return nw.from_numpy(
                np.column_stack([X_valid, weights]),
                schema=list(self.feature_names_in_) + ["weight"],
                backend="pandas",
            )
        else:
            return np.column_stack([X_valid, weights])

    def fit_transform(self, X, y, *, sensitive_features=None, **fit_params):
        """
        Fit the transformer and return the transformed data with weights.

        This is equivalent to calling ``fit`` followed by ``transform`` on
        the same data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        y : array-like of shape (n_samples,)
            Target labels.

        sensitive_features : iterable of int or str, optional
            Indices or names of sensitive feature columns.

        **fit_params : dict
            Additional fit parameters. These are accepted for API
            compatibility and are ignored.

        Returns
        -------
        X_transformed : array-like or DataFrame
            The input data augmented with an additional ``"weight"`` column.
        """
        return self.fit(X, y, sensitive_features=sensitive_features).transform(X, y)
