# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numbers
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import _multinomial_loss_grad, _logistic_loss_and_grad, _check_multi_class
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.utils import check_array, check_consistent_length, check_random_state, compute_class_weight
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.optimize import _check_optimize_result
from scipy import optimize


_LOGISTIC_SOLVER_CONVERGENCE_MSG = (
    "Please also refer to the documentation for alternative solver options:\n"
    "    https://scikit-learn.org/stable/modules/linear_model.html"
    "#logistic-regression"
)


# Some helper check functions
def _check_solver(solver, penalty, dual):
    all_solvers = ["lbfgs"]
    if solver not in all_solvers:
        raise ValueError(
            "Logistic Regression supports only solvers in %s, got %s."
            % (all_solvers, solver)
        )

    all_penalties = ["l1", "l2", "elasticnet", "none"]
    if penalty not in all_penalties:
        raise ValueError(
            "Logistic Regression supports only penalties in %s, got %s."
            % (all_penalties, penalty)
        )

    if solver not in ["liblinear", "saga"] and penalty not in ("l2", "none"):
        raise ValueError(
            "Solver %s supports only 'l2' or 'none' penalties, got %s penalty."
            % (solver, penalty)
        )
    if solver != "liblinear" and dual:
        raise ValueError(
            "Solver %s supports only dual=False, got dual=%s" % (solver, dual)
        )

    if penalty == "elasticnet" and solver != "saga":
        raise ValueError(
            "Only 'saga' solver supports elasticnet penalty, got solver={}.".format(
                solver
            )
        )

    if solver == "liblinear" and penalty == "none":
        raise ValueError("penalty='none' is not supported for the liblinear solver")

    return solver


def _logistic_regression_path(
    X,  # TODO: Does A need to be in here? --> Probably yes, since we need it in the constraints parameter later on.
        # This depends on whether I implement the constraints in the fit function, or here. Not yet sure what is best
    A,
    y,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    solver="lbfgs",
    coef=None,
    class_weight=None,
    dual=False,
    penalty="l2",
    intercept_scaling=1.0,
    multi_class="auto",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    sensitive_feature_ids=None,  # Names or ids of the sensitive features in X
    sensitive_features_to_cov_thresh=None,  # Covariance threshold that should be achieved for every sens feature, dict
):
    """TODO: add docsting"""
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty, dual)

    constraints = _get_constraint_list_cov(X, A, y, sensitive_feature_ids, sensitive_features_to_cov_thresh)

    # Preprocessing.
    if check_input:
        X = check_array(
            X,
            accept_sparse="csr",
            dtype=np.float64,
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        y = check_array(y, ensure_2d=False, dtype=None)
        check_consistent_length(X, y)
    _, n_features = X.shape

    classes = np.unique(y)
    random_state = check_random_state(random_state)

    multi_class = _check_multi_class(multi_class, solver, len(classes))
    if pos_class is None and multi_class != "multinomial":
        if classes.size > 2:
            raise ValueError("To fit OvR, use the pos_class argument")
        # np.unique(y) gives labels in sorted order.
        pos_class = classes[1]

    # If sample weights exist, convert them to array (support for lists)
    # and check length
    # Otherwise set them to 1 for all examples
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype, copy=True)

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == "multinomial":
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first. for the
    # multinomial case this is not necessary.
    if multi_class == "ovr":
        w0 = np.zeros(n_features + int(fit_intercept), dtype=X.dtype)
        mask_classes = np.array([-1, 1])
        mask = y == pos_class
        y_bin = np.ones(y.shape, dtype=X.dtype)
        y_bin[~mask] = -1.0
        # for compute_class_weight

        if class_weight == "balanced":
            class_weight_ = compute_class_weight(
                class_weight, classes=mask_classes, y=y_bin
            )
            sample_weight *= class_weight_[le.fit_transform(y_bin)]

    else:
        if solver not in ["sag", "saga"]:
            lbin = LabelBinarizer()
            Y_multi = lbin.fit_transform(y)
            if Y_multi.shape[1] == 1:
                Y_multi = np.hstack([1 - Y_multi, Y_multi])
        else:
            # SAG multinomial solver needs LabelEncoder, not LabelBinarizer
            le = LabelEncoder()
            Y_multi = le.fit_transform(y).astype(X.dtype, copy=False)

        w0 = np.zeros(
            (classes.size, n_features + int(fit_intercept)), order="F", dtype=X.dtype
        )

    if coef is not None:
        # it must work both giving the bias term and not
        if multi_class == "ovr":
            if coef.size not in (n_features, w0.size):
                raise ValueError(
                    "Initialization coef is of shape %d, expected shape %d or %d"
                    % (coef.size, n_features, w0.size)
                )
            w0[: coef.size] = coef
        else:
            # For binary problems coef.shape[0] should be 1, otherwise it
            # should be classes.size.
            n_classes = classes.size
            if n_classes == 2:
                n_classes = 1

            if coef.shape[0] != n_classes or coef.shape[1] not in (
                n_features,
                n_features + 1,
            ):
                raise ValueError(
                    "Initialization coef is of shape (%d, %d), expected "
                    "shape (%d, %d) or (%d, %d)"
                    % (
                        coef.shape[0],
                        coef.shape[1],
                        classes.size,
                        n_features,
                        classes.size,
                        n_features + 1,
                    )
                )

            if n_classes == 1:
                w0[0, : coef.shape[1]] = -coef
                w0[1, : coef.shape[1]] = coef
            else:
                w0[:, : coef.shape[1]] = coef

    if multi_class == "multinomial":
        # scipy.optimize.minimize and newton-cg accepts only
        # ravelled parameters.
        if solver in ["lbfgs", "newton-cg"]:
            w0 = w0.ravel()
        target = Y_multi
        if solver == "lbfgs":

            def func(x, *args):
                return _multinomial_loss_grad(x, *args)[0:2]

        # elif solver == "newton-cg":
        #
        #     def func(x, *args):
        #         return _multinomial_loss(x, *args)[0]
        #
        #     def grad(x, *args):
        #         return _multinomial_loss_grad(x, *args)[1]
        #
        #     hess = _multinomial_grad_hess
        warm_start_sag = {"coef": w0.T}
    else:
        target = y_bin
        if solver == "lbfgs":
            func = _logistic_loss_and_grad
        # elif solver == "newton-cg":
        #     func = _logistic_loss
        #
        #     def grad(x, *args):
        #         return _logistic_loss_and_grad(x, *args)[1]
        #
        #     hess = _logistic_grad_hess
        warm_start_sag = {"coef": np.expand_dims(w0, axis=1)}

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == "lbfgs":
            iprint = [-1, 50, 1, 100, 101][
                np.searchsorted(np.array([0, 1, 2, 3]), verbose)
            ]
            opt_res = optimize.minimize(
                func,
                w0,
                method="L-BFGS-B",
                jac=True,
                args=(X, target, 1.0 / C, sample_weight),
                options={"iprint": iprint, "gtol": tol, "maxiter": max_iter},
                # constraints=constraints, TODO: Implement constraints
            )
            n_iter_i = _check_optimize_result(
                solver,
                opt_res,
                max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
            )
            w0, loss = opt_res.x, opt_res.fun
        # elif solver == "newton-cg":
        #     args = (X, target, 1.0 / C, sample_weight)
        #     w0, n_iter_i = _newton_cg(
        #         hess, func, grad, w0, args=args, maxiter=max_iter, tol=tol
        #     )
        # elif solver == "liblinear":
        #     coef_, intercept_, n_iter_i, = _fit_liblinear(
        #         X,
        #         target,
        #         C,
        #         fit_intercept,
        #         intercept_scaling,
        #         None,
        #         penalty,
        #         dual,
        #         verbose,
        #         max_iter,
        #         tol,
        #         random_state,
        #         sample_weight=sample_weight,
        #     )
        #     if fit_intercept:
        #         w0 = np.concatenate([coef_.ravel(), intercept_])
        #     else:
        #         w0 = coef_.ravel()
        #
        # elif solver in ["sag", "saga"]:
        #     if multi_class == "multinomial":
        #         target = target.astype(X.dtype, copy=False)
        #         loss = "multinomial"
        #     else:
        #         loss = "log"
        #     # alpha is for L2-norm, beta is for L1-norm
        #     if penalty == "l1":
        #         alpha = 0.0
        #         beta = 1.0 / C
        #     elif penalty == "l2":
        #         alpha = 1.0 / C
        #         beta = 0.0
        #     else:  # Elastic-Net penalty
        #         alpha = (1.0 / C) * (1 - l1_ratio)
        #         beta = (1.0 / C) * l1_ratio
        #
        #     w0, n_iter_i, warm_start_sag = sag_solver(
        #         X,
        #         target,
        #         sample_weight,
        #         loss,
        #         alpha,
        #         beta,
        #         max_iter,
        #         tol,
        #         verbose,
        #         random_state,
        #         False,
        #         max_squared_sum,
        #         warm_start_sag,
        #         is_saga=(solver == "saga"),
        #     )

        else:
            raise ValueError(
                "solver must be {'lbfgs'}, got '%s' instead" % solver
            )
            # raise ValueError(
            #     "solver must be one of {'liblinear', 'lbfgs', "
            #     "'newton-cg', 'sag'}, got '%s' instead" % solver
            # )

        if multi_class == "multinomial":
            n_classes = max(2, classes.size)
            multi_w0 = np.reshape(w0, (n_classes, -1))
            if n_classes == 2:
                multi_w0 = multi_w0[1][np.newaxis, :]
            coefs.append(multi_w0.copy())
        else:
            coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter


def _ohe_sensitive_features(X, sensitive_feature_ids):
    """
    One-hot-encode the sensitive features such that they can be splitted
    from X and that the constraints can be correctly coded per feature value.
    """
    enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary')
    if isinstance(X, pd.DataFrame):
        transformed = enc.fit_transform(X[sensitive_feature_ids]).toarray()
        # Create a Pandas DataFrame of the hot encoded column
        ohe_df = pd.DataFrame(transformed, columns=enc.get_feature_names_out())
        # concat with original data
        X = pd.concat([X, ohe_df], axis=1).drop(sensitive_feature_ids, axis=1)
        sensitive_feature_ids = list(enc.get_feature_names_out())
    else:  # Numpy array
        transformed = enc.fit_transform(X[:, sensitive_feature_ids]).toarray()
        if len(enc.get_feature_names_out()) == 1:  # Only a single binary column
            X[:, sensitive_feature_ids] = transformed
        else:  # One or more columns that contain at least three values
            # Delete the old column and append the transformed columns
            X_without_sensitive = np.delete(X, sensitive_feature_ids, axis=1)
            X_with_sensitive = np.append(X_without_sensitive, transformed, axis=1)
            # Need to return the new transformed sensitive feature ids, since they have changed
            sensitive_feature_ids = list(range(X_without_sensitive.shape[1], X_with_sensitive.shape[1]))
    return X, sensitive_feature_ids


class FairLogisticRegression(LogisticRegression):
    """" TODO: add docstring, check in BalancedRandomForestClassifier how they handle docstrings for inherited class"""

    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        super().__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio,
        )

    # Below code is almost entirely reused from the CorrelationRemover, should this maybe be abstracted higher up?
    # Also not sure if this should be a function of the FairLogisticRegression class, doesn't really feel like it.
    # That is also why I feel like it is a good reason to abstract it higher up. Seems like something for utils maybe?
    def _split_X(self, X, sensitive_feature_ids):
        """Split up X into a sensitive and non-sensitive group."""
        sensitive = [self.lookup_[i] for i in sensitive_feature_ids]
        non_sensitive = [i for i in range(X.shape[1]) if i not in sensitive]
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, non_sensitive], X.iloc[:, sensitive]
        else:  # Numpy arrays
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

    def fit(self, X, y, sensitive_feature_ids, sensitive_attrs_to_cov_thresh, sample_weight=None):
        """" TODO: add docstring"""
        # One-hot-encode the data and return the new sensitive feature ids that come along with the encoded data
        X_ohe, sensitive_feature_ids = _ohe_sensitive_features(X, sensitive_feature_ids)
        # Split the data similarly to how the CorrelationRemover does it
        self._create_lookup(X_ohe)
        X_nonsensitive, X_sensitive = self._split_X(X_ohe, sensitive_feature_ids)

        constraints = _get_constraint_list_cov(X_nonsensitive, X_sensitive, y,
                                               sensitive_feature_ids, sensitive_attrs_to_cov_thresh)

        # TODO: Think about whether the constraints should be implemented in `fit`, or in `_logistic_regression_path`

        return X_nonsensitive, X_sensitive
