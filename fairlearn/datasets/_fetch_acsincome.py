# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib

from sklearn.datasets import fetch_openml
from ._constants import _DOWNLOAD_DIRECTORY_NAME


def fetch_acsincome(*, cache=True, data_home=None,
                    as_frame=False, return_X_y=False):
    """Load the ACS Income dataset.
    
    Download it if necessary.

    ==============   ==============
    Samples total           1664500
    Dimensionality               12
    Features                   real
    Target                     real
    ==============   ==============

    Source: Paper: Frances Ding (2021) [1]_

    .. versionadded:: 0.8.0

    Parameters
    ----------
    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : str, default=None
        Specify another download and cache folder for the datasets.
        By default, all scikit-learn data is stored in '~/.fairlearn-data'
        subfolders.

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (1664500, 12)
            Each row corresponding to the 10 feature values in order as well as 
            the state code (ST) and the target varaible (PINCP).
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (1664500,)
            Integer denoting each person's income.
            A threshold can be applied as a postprocessing step to frame this
            as a binary classification problem.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 12
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the ACSIncome dataset.

    (data, target) : tuple of (numpy.ndarray, numpy.ndarray) or (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is False

    (data, target) : tuple of (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is True

    References
    ----------
    .. [1] Frances Ding, Moritz Hardt, John Miller, Ludwig Schmidt 
       "Retiring Adult: New Datasets for Fair Machine Learning"
      Advances in Neural Information Processing Systems 34, 2021.

    """
    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    return fetch_openml(
        data_id=43137,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
