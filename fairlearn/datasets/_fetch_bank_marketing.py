# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib

from sklearn.datasets import fetch_openml

from ._constants import _DOWNLOAD_DIRECTORY_NAME

_BANK_FEATURE_NAMES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


def fetch_bank_marketing(*, cache=True, data_home=None, as_frame=True, return_X_y=False):
    """Load the UCI bank marketing dataset (binary classification).

    Download it if necessary.

    ==============   ====================
    Samples total                   45211
    Dimensionality                     16
    Features         numeric, categorical
    Classes                             2
    ==============   ====================

    Source:

    - UCI Repository :footcite:`moro2012bank`
    - Paper: Moro et al. :footcite:`moro2014data`

    The data is related with direct marketing campaigns of a Portuguese
    banking institution. The marketing campaigns were based on phone calls.
    Often, more than one contact to the same client was required,
    in order to access if the product (bank term deposit) would be (or not) subscribed.

    The classification goal is to predict if the client will subscribe a
    term deposit (variable y).

    .. versionadded:: 0.5.0

    Parameters
    ----------
    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : str, default=None
        Specify another download and cache folder for the datasets.
        By default, all fairlearn data is stored in '~/.fairlearn-data'
        subfolders.

    as_frame : bool, default=True
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

        .. versionchanged:: 0.9.0
            Default value changed to True.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (45211, 16)
            Each row corresponding to the 16 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (45211,)
            Each value represents whether the client subscribed a
            term deposit which is 'yes' if the client subscribed and 'no'
            otherwise.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 16
            The ordered UCI column names: ``age``, ``job``, ``marital``,
            ``education``, ``default``, ``balance``, ``housing``, ``loan``,
            ``contact``, ``day``, ``month``, ``duration``, ``campaign``,
            ``pdays``, ``previous``, ``poutcome``. When ``as_frame`` is True the
            same names are used for the ``data`` and ``frame`` columns, and for
            the ``X`` returned by ``return_X_y=True``.

            .. versionchanged:: 0.15.0
                Previously these were whatever names the underlying OpenML
                record carried, which are placeholders (``V1`` ... ``V16``)
                rather than the UCI column names.
        DESCR : string
            Description of the UCI bank marketing dataset.
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
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    # For data_home see
    # https://github.com/scikit-learn/scikit-learn/issues/27447
    result = fetch_openml(
        data_id=1461,
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
    if observed_n_features != len(_BANK_FEATURE_NAMES):
        raise ValueError(
            f"Expected the OpenML record with data_id=1461 backing the UCI bank marketing "
            f"dataset to have {len(_BANK_FEATURE_NAMES)} features, but it has "
            f"{observed_n_features}. The authoritative feature names are applied "
            f"positionally and cannot be matched to this record."
        )

    if return_X_y:
        if as_frame:
            X = X.copy()
            X.columns = _BANK_FEATURE_NAMES
        return X, y
    result.feature_names = _BANK_FEATURE_NAMES
    if as_frame:
        result.data.columns = _BANK_FEATURE_NAMES
        result.frame.columns = _BANK_FEATURE_NAMES + [result.target.name]
    elif isinstance(getattr(result, "categories", None), dict):
        feature_name_map = dict(zip(original_feature_names, _BANK_FEATURE_NAMES, strict=True))
        result.categories = {
            feature_name_map.get(name, name): values for name, values in result.categories.items()
        }
    return result
