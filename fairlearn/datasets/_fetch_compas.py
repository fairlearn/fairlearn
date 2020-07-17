# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib
import warnings

from sklearn.datasets import fetch_openml
from ._constants import _DOWNLOAD_DIRECTORY_NAME

from fairlearn.exceptions import DataFairnessWarning


def fetch_compas(*, cache=True, data_home=None,
                 as_frame=False, return_X_y=False, warn=True):
    """Load the ProPublica COMPAS dataset (binary classification).

    Download it if necessary.

    ==============   ===================
    Samples total                   5278
    Dimensionality                    13
    Features         numeric and nominal
    Classes                            2
    ==============   ===================

    See [1]_ for the original dataset and [2]_ for pre-processing steps of the current
    dataset.

    The prediction task is to determine whether a defendant will be charged again within two
    years, which serves as a proxy for recidivism.

    **This dataset has known fairness issues**. Fairness-unaware classification models are at risk
    of incorrectly judging African-American defendants to be at a higher risk of recidivism and
    incorrectly judging Caucasian defendants to be at lower risk of recidivism. See [3]_ for
    more details.

    Here's a table of all the variables in order:

    =====================  ======================================================================
    sex                    sex of the defendant; 1 if Male, 0 if Female
    age                    age of the defendant
    juv_fel_count          number of juvenile felony convictions
    juv_misd_count         number of juvenile misdemeanor convictions
    juv_other_count        number of other juvenile convictions
    priors_count           number of prior convictions
    age_cat_25-45          1 if defendant's age >=25 and <45, otherwise 0
    age_cat_Greaterthan45  1 if defendant's age >=45, otherwise 0
    age_cat_Lessthan25     1 if defendant's age <25, otherwise 0
    race_African-American  1 if defendant's race African-American, otherwise 0
    race_Caucasian         1 if defendant's race Caucasian, otherwise 0
    c_charge_degree_F      1 if charged for felony, otherwise 0
    c_charge_degree_M      1 if charged for misdemeanor, otherwise 0
    =====================  ======================================================================

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

        data : ndarray, shape (5278, 13)
            Each row corresponding to the 13 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (5278,)
            Each value indicates whether a defendant was charged again within two years after the
            current charge.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 13
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the COMPAS dataset.

    (data, target) : tuple of (numpy.ndarray, numpy.ndarray) or (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is False

    (data, target) : tuple of (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is True

    See Also
    --------
    fairlearn.datasets.fetch_boston: Boston housing dataset (regression)
    fairlearn.datasets.fetch_adult: UCI Adult dataset (binary classification)

    Notes
    -----
    This dataset consists of 5278 samples and 13 features.

    - During pre-processing only instances of defendants of the two most common races were kept,
      which means that `race_African-American` and `race_Caucasian` are perfectly correlated.
    - During pre-processing charges for ordinary traffic offenses were removed, which means that
      c_charge_degree_F` and `c_charge_degree_M` are perfectly correlated.
    - The features `age_cat_25-45`, `age_cat_Greaterthan45` and `age_cat_Lessthan25` are included
      to allow linear classifiers to be non-linear in age.


    References
    ----------
    .. [1] https://github.com/propublica/compas-analysis/
    .. [2] https://www.openml.org/d/42193
    .. [3] https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm/

    """
    if warn:
        msg = "You are about to use a dataset with known fairness issues."
        warnings.warn(DataFairnessWarning(msg))
    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME
    return fetch_openml(
        data_id=42193,
        data_home=data_home,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
