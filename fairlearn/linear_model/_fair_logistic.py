# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numbers
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from joblib import Parallel

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import (
    _multinomial_loss,
    _logistic_loss,
    _check_multi_class,
)
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.utils import (
    check_array,
    check_consistent_length,
    check_random_state,
    compute_class_weight,
)
from sklearn.utils.extmath import row_norms
from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import _check_sample_weight

_LOGISTIC_SOLVER_CONVERGENCE_MSG = (
    "Please also refer to the documentation for alternative solver options:\n"
    "    https://scikit-learn.org/stable/modules/linear_model.html"
    "#logistic-regression"
)


# Some helper check functions
def _check_solver(solver, penalty):
    all_solvers = ["SLSQP"]
    if solver not in all_solvers:
        raise ValueError(
            "Fair Logistic Regression supports only solvers in %s, got %s."
            % (all_solvers, solver)
        )

    all_penalties = ["l1", "l2", "elasticnet", "none"]
    if penalty not in all_penalties:
        raise ValueError(
            "Logistic Regression supports only penalties in %s, got %s."
            % (all_penalties, penalty)
        )

    return solver


def _add_intercept(X):
    """Copied from the paper:
    https://github.com/mbilalzafar/fair-classification/blob/master/fair_classification/utils.py#L271-L276
    Add intercept to the data before linear classification"""
    m, n = X.shape
    intercept = np.ones(m).reshape(m, 1)  # the constant b
    return np.concatenate((X, intercept), axis=1)


def _log_logistic(X):
    """Copied from the paper:
    https://github.com/mbilalzafar/fair-classification/blob/master/fair_classification/loss_funcs.py#L59-L93
    This function is used from scikit-learn source code. Source link below"""

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0
    
    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function
    
    Returns
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    -----
    
    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    if X.ndim > 1:
        raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X)  # same dimensions and data types

    idx = X > 0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out


def _test_sensitive_attr_constraint_cov(
    model, x_arr, y_arr_dist_boundary, x_control, thresh
):
    """
    Copied from the paper:
    https://github.com/mbilalzafar/fair-classification/blob/master/fair_classification/utils.py#L348-L388
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distance from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr

    this function will return -1 if the constraint specified by thresh parameter is not satisfied
    otherwise it will return +1
    if the return value is >=0, then the constraint is satisfied
    """

    assert x_arr.shape[0] == x_control.shape[0]
    if (
        len(x_control.shape) > 1
    ):  # make sure we just have one column in the array
        assert x_control.shape[1] == 1

    arr = []
    if model is None:
        arr = y_arr_dist_boundary  # simply the output labels
    else:
        arr = np.dot(
            model, x_arr.T
        )  # the product with the weight vector -- the sign of this is the output label

    arr = np.array(arr, dtype=np.float64)

    cov = np.dot(x_control - np.mean(x_control), arr) / float(len(x_control))

    ans = thresh - abs(
        cov
    )  # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied

    return ans


def _get_constraint_list_cov(
    X_train,
    A_train,
    y_train,
    renamed_sensitive_feature_ids,
    sensitive_attrs_to_cov_thresh,
):
    # For now, I only work with a constant threshold.
    # The paper works with a threshold per category of a sensitive feature (if not binary), and else per sens feature.
    constraints = []

    for attr in renamed_sensitive_feature_ids:
        if isinstance(A_train, pd.DataFrame):
            c = {
                "type": "ineq",
                "fun": _test_sensitive_attr_constraint_cov,
                "args": (
                    X_train,
                    y_train,
                    A_train[attr].to_numpy(),
                    sensitive_attrs_to_cov_thresh,
                ),
            }
        else:
            c = {
                "type": "ineq",
                "fun": _test_sensitive_attr_constraint_cov,
                "args": (
                    X_train,
                    y_train,
                    A_train[:, attr],
                    sensitive_attrs_to_cov_thresh,
                ),
            }

        constraints.append(c)
    return constraints


def _logistic_regression_path(
    X,
    y,
    pos_class=None,
    Cs=10,
    fit_intercept=True,
    max_iter=100,
    tol=1e-4,
    verbose=0,
    solver="SLSQP",
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
    l1_ratio=None,
    constraints=None,
):
    """
    TODO: add docstring
    All this code is copied from the sklearn _logistic_regression_path function, except for the constraints argument,
    the changed solver and the loss function used (all in the optimize.minimize function)
    """
    if isinstance(Cs, numbers.Integral):
        Cs = np.logspace(-4, 4, Cs)

    solver = _check_solver(solver, penalty)

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
    sample_weight = _check_sample_weight(
        sample_weight, X, dtype=X.dtype, copy=True
    )

    # If class_weights is a dict (provided by the user), the weights
    # are assigned to the original labels. If it is "balanced", then
    # the class_weights are assigned after masking the labels with a OvR.
    le = LabelEncoder()
    if isinstance(class_weight, dict) or multi_class == "multinomial":
        class_weight_ = compute_class_weight(
            class_weight, classes=classes, y=y
        )
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
            (classes.size, n_features + int(fit_intercept)),
            order="F",
            dtype=X.dtype,
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

    # TODO: Sklearn uses different loss functions based on multinomial or ovr, this is currently not implemented
    if multi_class == "multinomial":
        # scipy.optimize.minimize (used in SLSQP) accepts only
        # ravelled parameters.
        w0 = w0.ravel()
        target = Y_multi

        def func(x, *args):
            return _multinomial_loss(x, *args)[0]

    else:
        target = y_bin
        func = _logistic_loss

    coefs = list()
    n_iter = np.zeros(len(Cs), dtype=np.int32)
    for i, C in enumerate(Cs):
        if solver == "SLSQP":
            iprint = [-1, 50, 1, 100, 101][
                np.searchsorted(np.array([0, 1, 2, 3]), verbose)
            ]
            opt_res = optimize.minimize(
                fun=func,
                x0=w0,
                method="SLSQP",
                # jac=True,
                args=(X, target, 1.0 / C, sample_weight),
                options={"iprint": iprint, "maxiter": max_iter},
                constraints=constraints,
            )
            n_iter_i = _check_optimize_result(
                solver="lbfgs",  # Not the actual solver we are using, but it works fine like this
                result=opt_res,
                max_iter=max_iter,
                extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
            )
            w0, loss = opt_res.x, opt_res.fun

        else:
            raise ValueError(
                "solver must be {'SLSQP'}, got '%s' instead" % solver
            )

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
    enc = OneHotEncoder(handle_unknown="ignore")
    if isinstance(X, pd.DataFrame):
        transformed = enc.fit_transform(X[sensitive_feature_ids]).toarray()
        renamed_sensitive_feature_ids = list(enc.get_feature_names_out())
        # Create a Pandas DataFrame of the hot encoded column
        ohe_df = pd.DataFrame(
            transformed, columns=renamed_sensitive_feature_ids
        )
        # concat with original data, drop the original sensitive_feature_ids
        X = pd.concat([X, ohe_df], axis=1).drop(sensitive_feature_ids, axis=1)
    else:  # Numpy array
        transformed = enc.fit_transform(X[:, sensitive_feature_ids]).toarray()
        # Delete the old column and append the transformed columns
        X_without_sensitive = np.delete(X, sensitive_feature_ids, axis=1)
        X = np.append(X_without_sensitive, transformed, axis=1)
        # Need to return the new transformed sensitive feature ids, since there are more columns now
        renamed_sensitive_feature_ids = list(
            range(X_without_sensitive.shape[1], X.shape[1])
        )

    return X, renamed_sensitive_feature_ids


class FairLogisticRegression(LogisticRegression):
    """TODO: add docstring, check in BalancedRandomForestClassifier how they handle docstrings for inherited class"""

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
        solver="SLSQP",
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
            return (
                X.iloc[:, non_sensitive],
                X.iloc[:, sensitive],
                sensitive_feature_ids,
            )
        else:  # Numpy arrays
            # Change the sensitive_feature_ids because they are now in a different array with different indices
            sensitive_feature_ids = list(range(X[:, sensitive].shape[1]))
            return X[:, non_sensitive], X[:, sensitive], sensitive_feature_ids

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

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        sensitive_feature_ids=None,
        sensitive_attrs_to_cov_thresh=None,
    ):
        """TODO: add docstring"""
        # TODO: Maybe turn below code until constraints into a preprocessing function?

        # Probably shouldn't do it like this, this is kind of how the paper does it.
        # I should be looking to implement it more like sklearn does.
        if self.fit_intercept:
            X = _add_intercept(X)
        # One-hot-encode the data and return the new sensitive feature ids that come along with the encoded data
        X_ohe, renamed_sensitive_feature_ids = _ohe_sensitive_features(
            X, sensitive_feature_ids
        )
        # Split the data similarly to how the CorrelationRemover does it
        self._create_lookup(X_ohe)
        (
            X_nonsensitive,
            X_sensitive,
            renamed_sensitive_feature_ids,
        ) = self._split_X(X_ohe, renamed_sensitive_feature_ids)

        # TODO: Think about whether the constraints should be implemented in `fit`, or in `_logistic_regression_path`
        constraints = _get_constraint_list_cov(
            X_nonsensitive,
            X_sensitive,
            y,
            renamed_sensitive_feature_ids,
            sensitive_attrs_to_cov_thresh,
        )

        # We continue with the code from sklearn here
        solver = _check_solver(self.solver, self.penalty)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError(
                "Penalty term must be positive; got (C=%r)" % self.C
            )
        if self.penalty == "elasticnet":
            if (
                not isinstance(self.l1_ratio, numbers.Number)
                or self.l1_ratio < 0
                or self.l1_ratio > 1
            ):
                raise ValueError(
                    "l1_ratio must be between 0 and 1; got (l1_ratio=%r)"
                    % self.l1_ratio
                )
        elif self.l1_ratio is not None:
            warnings.warn(
                "l1_ratio parameter is only used when penalty is "
                "'elasticnet'. Got "
                "(penalty={})".format(self.penalty)
            )
        if self.penalty == "none":
            if self.C != 1.0:  # default values
                warnings.warn(
                    "Setting penalty='none' will ignore the C and l1_ratio parameters"
                )
                # Note that check for l1_ratio is done right above
            C_ = np.inf
            penalty = "l2"
        else:
            C_ = self.C
            penalty = self.penalty
        if not isinstance(self.max_iter, numbers.Number) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iteration must be positive; got (max_iter=%r)"
                % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be positive; got (tol=%r)"
                % self.tol
            )

        if solver == "SLSQP":
            _dtype = np.float64
        else:
            _dtype = [np.float64, np.float32]

        X_nonsensitive, y = self._validate_data(
            X_nonsensitive,
            y,
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        multi_class = _check_multi_class(
            self.multi_class, solver, len(self.classes_)
        )

        if solver in ["sag", "saga"]:
            max_squared_sum = row_norms(X, squared=True).max()
        else:
            max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r" % classes_[0]
            )

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, "coef_", None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(
                warm_start_coef, self.intercept_[:, np.newaxis], axis=1
            )

        # Hack so that we iterate only once for the multinomial case.
        if multi_class == "multinomial":
            classes_ = [None]
            warm_start_coef = [warm_start_coef]
        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        # The SAG solver releases the GIL so it's more efficient to use
        # threads for this solver.
        if solver in ["sag", "saga"]:
            prefer = "threads"
        else:
            prefer = "processes"
        fold_coefs_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer=prefer),
        )(
            path_func(
                X_nonsensitive,  # Only use the nonsensitive features
                y,
                pos_class=class_,
                Cs=[C_],
                l1_ratio=self.l1_ratio,
                fit_intercept=self.fit_intercept,
                tol=self.tol,
                verbose=self.verbose,
                solver=solver,
                multi_class=multi_class,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                check_input=False,
                random_state=self.random_state,
                coef=warm_start_coef_,
                penalty=penalty,
                max_squared_sum=max_squared_sum,
                sample_weight=sample_weight,
                constraints=constraints,
            )
            for class_, warm_start_coef_ in zip(classes_, warm_start_coef)
        )

        fold_coefs_, _, n_iter_ = zip(*fold_coefs_)
        self.n_iter_ = np.asarray(n_iter_, dtype=np.int32)[:, 0]

        if self.fit_intercept:
            n_features = (
                X_nonsensitive.shape[1] - 1
            )  # Don't count the intercept --> Probably shouldn't do it like this
        else:
            n_features = X_nonsensitive.shape[1]
        if multi_class == "multinomial":
            self.coef_ = fold_coefs_[0][0]
        else:
            self.coef_ = np.asarray(fold_coefs_)
            self.coef_ = self.coef_.reshape(
                n_classes, n_features + int(self.fit_intercept)
            )

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        else:
            self.intercept_ = np.zeros(n_classes)

        return self


if __name__ == "__main__":
    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=1590, as_frame=True)
    y_global = (data.target == ">50K") * 1
    FairLR = FairLogisticRegression(verbose=10)
    # Pandas test code
    # X_global = data.data[['age', 'fnlwgt', 'education-num', 'sex']]
    # Add another category to sex for extra testing
    # extra_cat = pd.DataFrame({"age": 25, "fnlwgt": 226802, "education-num": 7, "sex": "Unknown"},
    #                          index=[len(X_global)])
    # X_global = pd.concat([X_global, extra_cat])
    # X_global["sex2"] = X_global["sex"]
    # X_global.at[48842, "sex2"] = "Male"
    # y_global = y_global.append(pd.Series(1))
    # FairLR.fit(X_global, y_global, sensitive_feature_ids=['sex', 'sex2'], sensitive_attrs_to_cov_thresh=0)

    # Numpy arrays test code
    X_global = data.data[["age", "fnlwgt", "education-num", "sex"]].to_numpy()
    y_global = y_global.to_numpy()
    FairLR.fit(
        X_global,
        y_global,
        sensitive_feature_ids=[3],
        sensitive_attrs_to_cov_thresh=0,
    )

    # Predict on the first sample
    print(FairLR.predict([X_global[0, :3]]))
