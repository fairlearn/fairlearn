# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib
import warnings

from sklearn.datasets import fetch_openml
from ._constants import _DOWNLOAD_DIRECTORY_NAME

from fairlearn.exceptions import DataFairnessWarning


def fetch_boston(*, cache=True, data_home=None,
                 as_frame=False, return_X_y=False, warn=True):
    """Load the boston housing dataset (regression).

    Download it if necessary.

    ==============   ==============
    Samples total               506
    Dimensionality               13
    Features                   real
    Target           real 5. - 50.
    ==============   ==============

    This dataset has known fairness issues. There's a "lower status of
    population" (LSTAT) parameter that you need to look out for and a column
    that is a derived from the proportion of people with a black skin color
    that live in a neighborhood (B). See the references at the bottom for
    more detailed information.

    Here's a table of all the variables in order:

    =======  ======================================================================
    CRIM     per capita crime rate by town
    ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    INDUS    proportion of non-retail business acres per town
    CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    NOX      nitric oxides concentration (parts per 10 million)
    RM       average number of rooms per dwelling
    AGE      proportion of owner-occupied units built prior to 1940
    DIS      weighted distances to five Boston employment centres
    RAD      index of accessibility to radial highways
    TAX      full-value property-tax rate per $10,000
    PTRATIO  pupil-teacher ratio by town
    B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    LSTAT    % lower status of the population
    MEDV     Median value of owner-occupied homes in $1000's
    =======  ======================================================================


    Parameters
    ----------
    cache : boolean, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all fairlearn data is stored in '~/.fairlearn-data' subfolders.

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

    warn : boolean, default=True.
        If True, it raises an extra warning to make users aware of the unfairness
        aspect of this dataset.


    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (506, 13)
            Each row corresponding to the 13 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (506,)
            Each value corresponds to the average
            house value in units of 100,000.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 13
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the Boston housing dataset.

    (data, target) : tuple of (numpy.ndarray, numpy.ndarray) or (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is False

    (data, target) : tuple of (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is True

    Notes
    -----
    This dataset consists of 506 samples and 13 features. It is notorious for the fairness
    issues related to the `B` column. There's more information in the references.

    References
    ----------
    https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8
    https://github.com/scikit-learn/scikit-learn/issues/16155

    """
    if warn:
        msg = "You are about to use a dataset with known fairness issues."
        warnings.warn(DataFairnessWarning(msg))
    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME
    return fetch_openml(
        data_id=531,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
