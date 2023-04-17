# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib

from sklearn.datasets import fetch_openml

import fairlearn.utils._compatibility as compat

from ._constants import _DOWNLOAD_DIRECTORY_NAME


def fetch_adult(*, cache=True, data_home=None, as_frame=True, return_X_y=False):
    """Load the UCI Adult dataset (binary classification).

    Read more in the :ref:`User Guide <boston_housing_data>`.

    Download it if necessary.

    ==============   ====================
    Samples total                   48842
    Dimensionality                     14
    Features         numeric, categorical
    Classes                             2
    ==============   ====================

    Source: UCI Repository [1]_ , Paper: R. Kohavi (1996) [2]_

    Prediction task is to determine whether a person makes over $50,000 a
    year.

    Read more in the :ref:`User Guide <adult_data>`.

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

        data : ndarray, shape (48842, 14)
            Each row corresponding to the 14 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (48842,)
            Each value represents whether the person earns more than $50,000
            a year (>50K) or not (<=50K).
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 14
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the UCI Adult dataset.
        categories : dict or None
            Maps each categorical feature name to a list of values, such that the
            value encoded as i is ith in the list. If ``as_frame`` is True, this is None.
        frame : pandas DataFrame
            Only present when ``as_frame`` is True. DataFrame with ``data`` and ``target``.

    (data, target) : tuple if ``return_X_y`` is True

    Notes
    ----------
    Our API largely follows the API of :func:`sklearn.datasets.fetch_openml`.

    References
    ----------
    .. [1] R. Kohavi and B. Becker, UCI Machine Learning Repository:
       Adult Data Set, 01-May-1996. [Online]. Available:
       https://archive.ics.uci.edu/ml/datasets/adult.

    .. [2] R. Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers:
       a Decision-Tree Hybrid," in *Second International Conference on knowledge
       discovery and data mining: proceedings: August 2-4, 1996, Portland,
       Oregon*, 1996, pp. 202–207.

    """
    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    return fetch_openml(
        data_id=1590,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
        **compat._PARSER_KWARG,
    )
