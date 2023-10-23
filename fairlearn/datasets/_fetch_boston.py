# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib
import warnings

from sklearn.datasets import fetch_openml

from fairlearn.exceptions import DataFairnessWarning

from ._constants import _DOWNLOAD_DIRECTORY_NAME


def fetch_boston(
    *, cache=True, data_home=None, as_frame=True, return_X_y=False, warn=True
):
    """Load the boston housing dataset (regression).

    Download it if necessary.

    ==============   ==============
    Samples total               506
    Dimensionality               13
    Features                   real
    Target           real 5. - 50.
    ==============   ==============

    Source:

    - OpenML :footcite:`vanschoren2014boston`
    - Paper: Harrison and Rubinfeld :footcite:`harrison1978hedonic`

    The Boston house-price data of
    D. Harrison, and D.L. Rubinfeld :footcite:`harrison1978hedonic`.

    Referenced in Belsley, Kuh & Welsch :footcite:`belsley2005regression`.

    This dataset has known fairness issues :footcite:`sykes2020boston`.
    There's a "lower status of population" (LSTAT) parameter that you need
    to look out for and a column that is a derived from the proportion of
    people with a black skin color that live in a neighborhood (B)
    :footcite:`carlisle2019racist`.
    See the references at the bottom for  more detailed information.

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
    B        1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
    LSTAT    % lower status of the population
    MEDV     Median value of owner-occupied homes in $1000's
    =======  ======================================================================

     Read more in the :ref:`User Guide <boston_housing_data>`.

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

    warn : bool, default=True
        If True, it raises an extra warning to make users aware of the unfairness
        aspect of this dataset.


    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
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
        categories : dict or None
            Maps each categorical feature name to a list of values, such that the
            value encoded as i is ith in the list. If ``as_frame`` is True, this is None.
        frame : pandas DataFrame
            Only present when ``as_frame`` is True. DataFrame with ``data`` and ``target``.

    (data, target) : tuple if ``return_X_y`` is True

    Notes
    -----
    Our API largely follows the API of :func:`sklearn.datasets.fetch_openml`.
    This dataset consists of 506 samples and 13 features. It is notorious for the fairness
    issues related to the `B` column. There's more information in the references.

    References
    ----------
    .. footbibliography::

    """
    if warn:
        msg = "You are about to use a dataset with known fairness issues."
        warnings.warn(DataFairnessWarning(msg))
    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    # For data_home see
    # https://github.com/scikit-learn/scikit-learn/issues/27447
    return fetch_openml(
        data_id=531,
        data_home=str(data_home),
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
        parser="auto",
    )
