# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib

from sklearn.datasets import fetch_openml
from ._constants import _DOWNLOAD_DIRECTORY_NAME


def fetch_bank_marketing(*, cache=True, data_home=None,
                         as_frame=False, return_X_y=False):
    """Load the UCI bank marketing dataset (binary classification).

    Download it if necessary.

    ==============   ====================
    Samples total                   45211
    Dimensionality                     17
    Features         numeric, categorical
    Classes                             2
    ==============   ====================

    Source: http://archive.ics.uci.edu/ml/datasets/Bank+Marketing
    The data is related with direct marketing campaigns of a Portuguese
    banking institution. The marketing campaigns were based on phone calls.
    Often, more than one contact to the same client was required,
    in order to access if the product (bank term deposit) would be (or not) subscribed.

    The classification goal is to predict if the client will subscribe a
    term deposit (variable y).


    Parameters
    ----------
    cache : boolean, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/.fairlearn-data' subfolders.

    as_frame : boolean, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

    return_X_y : boolean, default=False.
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (45211, 17)
            Each row corresponding to the 17 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (45211,)
            Each value represents whether the client subscribed a
            term deposit which is 'yes' if the client subscribed and 'no'
            otherwise.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 17
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the UCI bank marketing dataset.

    (data, target) : tuple of (numpy.ndarray, numpy.ndarray) or (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is False

    (data, target) : tuple of (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is True

    """
    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    return fetch_openml(
        data_id=1461,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
