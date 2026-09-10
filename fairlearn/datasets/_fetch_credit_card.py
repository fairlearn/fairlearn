# Copyright (c) Fairlearn Contributors.
# Licensed under the MIT License.

from pathlib import Path

from sklearn.datasets import fetch_openml

from ._constants import _DOWNLOAD_DIRECTORY_NAME

_CREDIT_FEATURE_NAMES = (
    [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def fetch_credit_card(*, cache=True, data_home=None, as_frame=True, return_X_y=False):
    """Load the 'Default of Credit Card clients' dataset (binary classification).

    ===============   ===============
    Samples total               30000
    Dimensionality                 23
    Features                     real
    Classes                         2
    ===============   ===============

    Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    I-Cheng Yeh and Che-hui Lien, "The comparisons of data mining techniques for the
    predictive accuracy of probability of default of credit card clients", Expert Systems
    with Applications, 36(2), 2473-2480, 2009

    .. versionadded:: 0.9.0

    Parameters
    ----------
    cache : boolean, default=True
        Whether to cache downloaded datasets using joblib

    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default,
        all scikit-learn data is stored in '~/.fairlearn-data' subfolders.

    as_frame : boolean, default=True
        If True,
            Returns the data as Pandas DataFrame, and the target
            is returned as a Pandas Series.
        If False,
            Returns a scikit-learn Bunch object with ``frame`` attribute containing
            the data and the target.

        .. versionchanged:: 0.9.0
            Default value changed to True.

    return_X_y : boolean, default=False.
        If True,
            returns ``(data.data, data.target)``
        Else,
            return Sci-kit Learn Bunch object


    Returns
    -------
    dataset : class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : NumPy Array or Pandas DataFrame, Shape (30000, 23)
            Each row corresponds to the 23 feature values in order.
            If ``as_frame`` is True, ``data`` is a Pandas DataFrame
        target : NumPy Array or Pandas Series, Shape (30000,)
            Each value represents whether an applicant defaulted on credit loan.
            If ``as_frame`` is True, ``target`` is a Pandas Series.
        feature_names : List of Strings, Length 23
            The ordered UCI column names: ``LIMIT_BAL``, ``SEX``, ``EDUCATION``,
            ``MARRIAGE``, ``AGE``, ``PAY_0``, ``PAY_2`` ... ``PAY_6``,
            ``BILL_AMT1`` ... ``BILL_AMT6``, ``PAY_AMT1`` ... ``PAY_AMT6``.
            When ``as_frame`` is True the same names are used for the ``data``
            and ``frame`` columns, and for the ``X`` returned by
            ``return_X_y=True``.

            .. versionchanged:: 0.15.0
                Previously these were whatever names the underlying OpenML
                record carried, which are placeholders (``x1`` ... ``x23``)
                rather than the UCI column names.
        DESCR : string
            Description of the UCI Default of Credit Card
        categories : dict or None
            Maps each categorical feature name to a list of values, such that the
            value encoded as i is ith in the list. If ``as_frame`` is True, this is None.
            The keys use the UCI names listed under ``feature_names``.
        frame : pandas DataFrame
            Only present when ``as_frame`` is True. DataFrame with ``data`` and ``target``.

    (data, target) : tuple if ``return_X_y`` is True

    Notes
    -----
    Our API largely follows the API of :func:`sklearn.datasets.fetch_openml`.

    """
    if not data_home:
        data_home = Path().home() / _DOWNLOAD_DIRECTORY_NAME

    # For data_home see
    # https://github.com/scikit-learn/scikit-learn/issues/27447
    result = fetch_openml(
        data_id=42477,
        data_home=str(data_home),
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
        parser="auto",
    )
    # The authoritative names are applied positionally, so refuse to relabel a record
    # whose width does not match instead of silently mislabelling the columns.
    if return_X_y:
        X, y = result
        original_feature_names = list(X.columns) if as_frame else []
        observed_n_features = X.shape[1]
    else:
        original_feature_names = list(result.feature_names)
        observed_n_features = len(original_feature_names)
    if observed_n_features != len(_CREDIT_FEATURE_NAMES):
        raise ValueError(
            f"Expected the OpenML record with data_id=42477 backing the UCI 'Default of Credit Card clients' "
            f"dataset to have {len(_CREDIT_FEATURE_NAMES)} features, but it has "
            f"{observed_n_features}. The authoritative feature names are applied "
            f"positionally and cannot be matched to this record."
        )

    if return_X_y:
        if as_frame:
            X = X.copy()
            X.columns = _CREDIT_FEATURE_NAMES
        return X, y
    result.feature_names = _CREDIT_FEATURE_NAMES
    if as_frame:
        result.data.columns = _CREDIT_FEATURE_NAMES
        result.frame.columns = _CREDIT_FEATURE_NAMES + [result.target.name]
    elif isinstance(getattr(result, "categories", None), dict):
        feature_name_map = dict(zip(original_feature_names, _CREDIT_FEATURE_NAMES, strict=True))
        result.categories = {
            feature_name_map.get(name, name): values for name, values in result.categories.items()
        }
    return result
