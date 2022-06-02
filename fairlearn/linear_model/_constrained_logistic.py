# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import numbers
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from joblib import Parallel

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._logistic import (
    _logistic_loss,
)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import (
    check_array,
    check_consistent_length,
    compute_class_weight,
)
from sklearn.utils.fixes import delayed, _joblib_parallel_args
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.optimize import _check_optimize_result
from sklearn.utils.validation import _check_sample_weight


def _check_solver(solver, penalty):
    all_solvers = ["SLSQP"]
    if solver not in all_solvers:
        raise ValueError(
            "Constrained Logistic Regression supports only solvers in %s, got %s."
            % (all_solvers, solver)
        )

    all_penalties = ["l2", "none"]
    if penalty not in all_penalties:
        raise ValueError(
            "Constrained Logistic Regression supports only penalties in %s, got %s."
            % (all_penalties, penalty)
        )

    return solver


def _check_multi_class(multi_class):
    if multi_class not in ("ovr"):
        raise ValueError("multi_class should be 'ovr'. Got %s." % multi_class)
    return multi_class


def _sensitive_attr_constraint_cov(model, X, sensitive_features, covariance_bound):
    """
    Calculate the covariance threshold.

    Calculate the difference between the covariance threshold and the (absolute)
    covariance between the sensitive feature and the (signed) distance between the
    instances’ feature vectors and the decision boundary of the logistic regression
    model.

    Parameters
    ----------
    model : numpy.ndarray
        Model weights.

    X : numpy.ndarray or pandas.DataFrame
        Feature data.

    sensitive_features : numpy.ndarray or pandas.DataFrame
        Sensitive features.

    covariance_bound : float
        The covariance threshold, which specifies the upper bound on the (absolute)
        covariance.

    Returns
    -------
    ans : float
        The difference between the given covariance_threshold and the absolute
        covariance.
    """
    if X.shape[0] != sensitive_features.shape[0]:
        raise ValueError(
            f"X has {X.shape[0]} instances while"
            " the sensitive features consist of"
            f" {sensitive_features.shape[0]} instances."
            " These should be equal."
        )

    intercept = 0.0

    num_features = X.shape[1]
    if model.size == num_features + 1:  # True if the intercept needs to be fitted
        intercept = model[-1]
        model = model[:-1]

    # the product with the weight vector -- the sign of this is the output label
    arr = np.dot(model, X.T) + intercept

    arr = np.array(arr, dtype=np.float64)
    cov = np.dot(sensitive_features - np.mean(sensitive_features), arr) / float(
        len(sensitive_features)
    )
    ans = covariance_bound - abs(cov)
    return ans


def _get_constraint_list_cov(
    X,
    sensitive_features,
    categories,
    covariance_bound,
):
    """
    Collect all constraints in a list.

    Parameters
    ----------
    X : numpy.ndarray or pandas.DataFrame
        Feature data.

    sensitive_features : numpy.ndarray or pandas.DataFrame
        Sensitive features.

    categories : list
        The categories from the one-hot-encoder.

    covariance_bound : float
        The given covariance threshold that we optimize towards.

    Returns
    -------
    constraints : list
        A list of the correctly formatted constraints, one for each sensitive feature.
    """
    # New covariance bound list such that it correctly handles the OHE sensitive features
    # Now we can have a separate covariance bound per sensitive feature
    cov_bound = []
    for index, array in enumerate(categories):
        cov_bound.extend([covariance_bound[index]] * len(array))

    constraints = []

    if isinstance(sensitive_features, pd.DataFrame):
        to_iterate = list(sensitive_features.columns)
    else:  # Numpy array
        to_iterate = list(range(sensitive_features.shape[1]))

    for index, attr in enumerate(to_iterate):
        if isinstance(sensitive_features, pd.DataFrame):
            c = {
                "type": "ineq",
                "fun": _sensitive_attr_constraint_cov,
                "args": (
                    X,
                    sensitive_features[attr].to_numpy(),
                    cov_bound[index],
                ),
            }
        else:
            c = {
                "type": "ineq",
                "fun": _sensitive_attr_constraint_cov,
                "args": (
                    X,
                    sensitive_features[:, attr],
                    cov_bound[index],
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
    multi_class="ovr",
    random_state=None,
    check_input=True,
    max_squared_sum=None,
    sample_weight=None,
    l1_ratio=None,
    constraints=None,
):
    """
    Compute a Logistic Regression model.

    All code comes from the sklearn _logistic_regression_path function,
    except for the constraints argument and the changed solver
    (all in the optimize.minimize function). Some parts of the original
    code in sklearn might be removed, since they are not relevant here.
    For example, we have only "ovr" implemented as multi_class
    option, meaning that we don't need code for the multinomial case.
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

    multi_class = _check_multi_class(multi_class)
    if pos_class is None:
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
    if isinstance(class_weight, dict):
        class_weight_ = compute_class_weight(class_weight, classes=classes, y=y)
        sample_weight *= class_weight_[le.fit_transform(y)]

    # For doing a ovr, we need to mask the labels first.
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

    if coef is not None:
        # it must work both giving the bias term and not
        if coef.size not in (n_features, w0.size):
            raise ValueError(
                "Initialization coef is of shape %d, expected shape %d or %d"
                % (coef.size, n_features, w0.size)
            )
        w0[: coef.size] = coef

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
                # jac=True,  # Commented because it is not used by the authors of the paper
                args=(X, target, 1.0 / C, sample_weight),
                options={"iprint": iprint, "maxiter": max_iter},
                constraints=constraints,
            )
            n_iter_i = _check_optimize_result(
                solver="lbfgs",  # Not the actual solver we are using, but it works fine like this
                result=opt_res,
                max_iter=max_iter,
            )
            w0, _ = opt_res.x, opt_res.fun

        else:
            raise ValueError("solver must be {'SLSQP'}, got '%s' instead" % solver)

        coefs.append(w0.copy())

        n_iter[i] = n_iter_i

    return np.array(coefs), np.array(Cs), n_iter


def _ohe_sensitive_features(sensitive_features):
    """
    One-hot-encode the sensitive features.

    We one-hot-encode the sensitive features such that the constraints
    can be correctly coded per feature value.

    Parameters
    ----------
    sensitive_features : numpy.ndarray or pandas.DataFrame
        Sensitive feature data.

    Returns
    -------
    transformed : numpy.ndarray or pandas.DataFrame
        Transformed sensitive feature data with one-hot-encoded values.

    categories : list
        The categories from the encoder.
    """
    enc = OneHotEncoder(handle_unknown="ignore")
    if isinstance(sensitive_features, pd.DataFrame):
        transformed = enc.fit_transform(sensitive_features).toarray()
        renamed_sensitive_feature = list(enc.get_feature_names_out())
        # Create a Pandas DataFrame of the hot encoded column
        transformed = pd.DataFrame(transformed, columns=renamed_sensitive_feature)

    else:  # Numpy array
        if len(sensitive_features.shape) == 1:  # 1-d array
            sensitive_features = sensitive_features.reshape(-1, 1)
        transformed = enc.fit_transform(sensitive_features).toarray()

    categories = enc.categories_

    return transformed, categories


def _process_covariance_bound_dict(covariance_bound, sensitive_features):
    if not isinstance(sensitive_features, pd.DataFrame):
        raise ValueError(
            f"The sensitive features are of type {type(sensitive_features)},"
            " while they should be supplied as a pd.DataFrame since the"
            " covariance bound values are supplied as a dictionary."
        )

    if not sorted(sensitive_features.columns) == sorted(covariance_bound.keys()):
        raise ValueError(
            "The keys in the covariance bound dictionary do not match"
            " the column names in the sensitive features DataFrame."
            f" Covariance bound keys: {covariance_bound.keys()},"
            f" sensitive features column names: {sensitive_features.columns}."
        )

    covariance_bound_list = []
    for column in sensitive_features.columns:
        covariance_bound_list.append(covariance_bound[column])

    return covariance_bound_list


def _process_sensitive_features(sensitive_features):
    """
    Process the sensitive features.

    We process the sensitive features such that we obtain the number
    of sensitive features that are supplied, as well as making sure
    that the sensitive features are stored in either a numpy array
    or a Pandas DataFrame. When the sensitive features are supplied as
    a list, we assume there is only one sensitive feature, and it is
    transformed to a numpy array. Sensitive features supplied via a
    pandas.Series are transformed to a pandas.DataFrame.

    Parameters
    ----------
    sensitive_features : List, pandas.Series, numpy.ndarray, pandas.DataFrame
        The sensitive features. In the case of a list or pandas.Series,
        we assume there is only one sensitive feature.

    Returns
    -------
    sensitive_features : numpy.ndarray or pandas.DataFrame
        Transformed sensitive feature data.

    count : int
        The number of sensitive features supplied.
    """
    if isinstance(sensitive_features, list):
        sensitive_features = np.atleast_1d(np.squeeze(np.asarray(sensitive_features)))
        assert len(sensitive_features.shape) == 1  # Sanity check
        count = len(sensitive_features.shape)
    elif isinstance(sensitive_features, pd.Series):
        count = 1  # A series always represents a single sensitive feature
        sensitive_features = pd.DataFrame(sensitive_features)
    elif isinstance(sensitive_features, np.ndarray):
        if len(sensitive_features.shape) > 1:  # If not a 1-d array
            count = sensitive_features.shape[1]
        else:
            count = len(sensitive_features.shape)  # This should always be 1
    elif isinstance(sensitive_features, pd.DataFrame):
        count = len(sensitive_features.columns)
    else:
        raise TypeError(
            "Sensitive features is of the wrong type."
            f" Got {type(sensitive_features)}, but was expecting one of"
            " (list, pd.Series, np.ndarrray, pd.DataFrame)."
        )
    return sensitive_features, count


class ConstrainedLogisticRegression(LogisticRegression):
    r"""Logistic Regression subject to a fairness constraint.

    This implementation closely follows the implementation in
    :class:`sklearn.linear_model.LogisticRegression`.

    Parameters
    ----------
    constraints : str, {'demographic_parity_covariance', none'},
        default='demographic_parity_covariance'
        The constraints used in the logistic regression:

        - `'demographic_parity_covariance'`:
            This constraint is defined as the covariance between the sensitive
            features :math:`\left\{\mathbf{z}_{i}\right\}_{i=1}^{N}`, and the
            signed distance from the users` feature vectors to the decision boundary
            :math:`\left\{d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right)\right\}_{i=1}^{N}`.
            This is mathemetically described as follows:

            .. math::
                \operatorname{Cov}\left(\mathbf{z}, d_{\boldsymbol{\theta}}(\mathbf{x})\right)
                =\mathbb{E}\left[(\mathbf{z}-\overline{\mathbf{z}})
                d_{\boldsymbol{\theta}}(\mathbf{x})\right]
                -\mathbb{E}[(\mathbf{z}-\overline{\mathbf{z}})]
                \bar{d}_{\boldsymbol{\theta}}(\mathbf{x})\\
                \approx \frac{1}{N} \sum_{i=1}^{N}\left(\mathbf{z}_{i}-\overline{\mathbf{z}}\right)
                d_{\boldsymbol{\theta}}\left(\mathbf{x}_{i}\right)

            The reasoning behind this derivation is further explained in
            :footcite:`zafar2017fairness`.
        - `'none'`: No constraint is used, resulting in a default sklearn
          `LogisticRegression`.

    penalty : str, {'l2', 'none'}, default='l2'
        Specify the norm of the penalty:

        - `'none'`: no penalty is added;
        - `'l2'`: add a L2 penalty term and it is the default choice;

    covariance_bound : float or list or dict, default=0
        The covariance bound that the constraint optimizes towards.
        It can either be one of the following two:

        - A single float that will be used for all sensitive features.
        - A list of floats such that each sensitive feature has its own covariance bound.
          The order in the list corresponds to the order of the passed sensitive features.
        - A dictionary containing the column names and covariance bound values as
          key-value pairs in the case when the sensitive features are supplied in a
          Pandas DataFrame.

    dual : bool, default=False
        Dual or primal formulation. Dual formulation is currently not implemented.
        Only implemented for compatibility with sklearn.

    tol : float, default=1e-4
        Tolerance for stopping criteria.

    C : float, default=1.0
        Inverse of regularization strength; must be a positive float.
        Like in support vector machines, smaller values specify stronger
        regularization.

    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float, default=1
        Dual formulation is currently not implemented.
        Only implemented for compatibility with sklearn.

    class_weight : dict or 'balanced', default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    random_state : int, RandomState instance, default=None
        Dual formulation is currently not implemented.
        Only implemented for compatibility with sklearn.

    solver : str, default='SLSQP'
        Algorithm to use in the optimization problem.
        SLSQP is used since it works with constraints. Other (default) solvers
        in sklearn are not compatible with constraints.

    max_iter : int, default=100
        Maximum number of iterations taken for the solvers to converge.

    multi_class : {'ovr'}, default='ovr'
        If the option chosen is 'ovr', then a binary problem is fit for each
        label. Sklearn also provides a 'multinomial' option, but this is not
        compatible with the constraints.

    verbose : int, default=0
        Set verbose to any positive number for verbosity.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See `the glossary <https://scikit-learn.org/stable/glossary.html#term-warm_start>`.

    n_jobs : int, default=None
        Number of CPU cores used when parallelizing over classes if
        multi_class='ovr'". ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors.
        See `the glossary <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`
        for more details.

    l1_ratio : float, default=None
        Dual formulation is currently not implemented.
        Only implemented for compatibility with sklearn.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.
        `coef_` is of shape (1, n_features) when the given problem is binary.
        In particular, when `multi_class='multinomial'`, `coef_` corresponds
        to outcome 1 (True) and `-coef_` corresponds to outcome 0 (False).

    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        If `fit_intercept` is set to False, the intercept is set to zero.
        `intercept_` is of shape (1,) when the given problem is binary.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_iter_ : ndarray of shape (n_classes,) or (1, )
        Actual number of iterations for all classes. If binary or multinomial,
        it returns only 1 element. For liblinear solver, only the maximum
        number of iteration across all classes is given.

    References
    ----------
    .. footbibliography::

    """

    def __init__(
        self,
        constraints="demographic_parity_covariance",
        penalty="l2",
        *,
        covariance_bound=0.0,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="SLSQP",
        max_iter=100,
        multi_class="ovr",
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

        self.constraints = constraints
        self.covariance_bound = covariance_bound

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        sensitive_features=None,
    ):
        """
        Fit the model according to the given training data and sensitive features.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        sensitive_features : list, pandas.Series, numpy.ndarray, pandas.DataFrame
            The sensitive features. In the case of a list or pandas.Series,
            we assume there is only one sensitive feature.

        Returns
        -------
        self
            Fitted estimator.

        Notes
        -----
        If `self.constraints` is None, the Logistic Regression from sklearn
        will be used with the parameters that you supplied.
        """
        if self.constraints is None:
            clf = LogisticRegression(
                penalty=self.penalty,
                dual=self.dual,
                tol=self.tol,
                C=self.C,
                fit_intercept=self.fit_intercept,
                intercept_scaling=self.intercept_scaling,
                class_weight=self.class_weight,
                random_state=self.random_state,
                solver="lbfgs",
                max_iter=self.max_iter,
                multi_class=self.multi_class,
                verbose=self.verbose,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                l1_ratio=self.l1_ratio,
            )
            return clf.fit(X, y)

        # The covariance_bound needs to be in a list for _get_constraint_list_cov
        if isinstance(self.covariance_bound, (float, int)):
            self.covariance_bound = [self.covariance_bound]
        elif isinstance(self.covariance_bound, dict):
            self.covariance_bound = _process_covariance_bound_dict(
                self.covariance_bound, sensitive_features
            )

        sensitive_features, num_sens_features = _process_sensitive_features(
            sensitive_features
        )

        if len(self.covariance_bound) > num_sens_features:
            raise ValueError(
                "Number of covariance bound values can not exceed"
                f" the amount of sensitive features. Got {len(self.covariance_bound)}"
                f" covariance bound values, got {num_sens_features} sensitive features."
            )

        if num_sens_features > len(self.covariance_bound):
            if len(self.covariance_bound) == 1:
                self.covariance_bound = self.covariance_bound * num_sens_features
            else:
                raise ValueError(
                    "Number of covariance bound values is higher than 1 but lower than"
                    " the amount of sensitive features. Got"
                    f" {len(self.covariance_bound)} covariance bound values, got"
                    f" {num_sens_features} sensitive features. Either pick a covariance"
                    " bound value per sensitive feature, or only one covariance bound"
                    " value."
                )

        sensitive_features, categories = _ohe_sensitive_features(sensitive_features)

        constraints = _get_constraint_list_cov(
            X,
            sensitive_features,
            categories,
            self.covariance_bound,
        )

        # We continue with the code from sklearn here
        solver = _check_solver(self.solver, self.penalty)

        if not isinstance(self.C, numbers.Number) or self.C < 0:
            raise ValueError("Penalty term must be positive; got (C=%r)" % self.C)
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

        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=_dtype,
            order="C",
            accept_large_sparse=solver not in ["liblinear", "sag", "saga"],
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)

        multi_class = _check_multi_class(self.multi_class)

        max_squared_sum = None

        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
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

        if warm_start_coef is None:
            warm_start_coef = [None] * n_classes

        path_func = delayed(_logistic_regression_path)

        prefer = "processes"
        fold_coefs_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **_joblib_parallel_args(prefer=prefer),
        )(
            path_func(
                X,
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

        n_features = X.shape[1]

        self.coef_ = np.asarray(fold_coefs_)
        self.coef_ = self.coef_.reshape(n_classes, n_features + int(self.fit_intercept))

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, -1]
            self.coef_ = self.coef_[:, :-1]
        else:
            self.intercept_ = np.zeros(n_classes)

        return self
