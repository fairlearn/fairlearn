# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import pathlib

from sklearn.datasets import fetch_openml

from ._constants import _DOWNLOAD_DIRECTORY_NAME


def fetch_diabetes_hospital(
    *, as_frame=True, cache=True, data_home=None, return_X_y=False
):
    """Load the preprocessed Diabetes 130-Hospitals dataset (binary classification).

    Download it if necessary.

    ==============   ============================
    Samples total                          101766
    Dimensionality                             24
    Features         numeric, categorical, string
    Classes                                     2
    ==============   ============================

    Source: UCI Repository :footcite:`strack2014diabetes`
    Paper: Strack et al., 2014 :footcite:`strack2014impact`

    The "Diabetes 130-Hospitals" dataset represents 10 years of clinical care at 130
    U.S. hospitals and delivery networks, collected from 1999 to 2008. Each record
    represents the hospital admission record for a patient diagnosed with diabetes
    whose stay lasted between one to fourteen days.

    The original "Diabetes 130-Hospitals" dataset was collected by Beata Strack,
    Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura,
    Krzysztof J. Cios, and John N. Clore in 2014.

    This version of the dataset was derived by the Fairlearn team for the SciPy
    2021 tutorial "Fairness in AI Systems: From social context to practice using
    Fairlearn". In this version, the target variable "readmitted" is binarized
    into whether the patient was re-admitted within thirty days. The full
    pre-processing script is available
    `here <https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/preprocess.py>`_.

    Read more in the :ref:`User Guide <diabetes_hospital_data>`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    as_frame : bool, default=True
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical).

        .. note::
            If set to False, this will raise an exception because of a type mismatch
            in the OpenML dataset.

        .. versionadded:: 0.9.0

    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : str, default=None
        Specify another download and cache folder for the datasets.
        By default, all fairlearn data is stored in '~/.fairlearn-data'
        subfolders.

    return_X_y : bool, default=False
        If True, returns ``(data.data, data.target)`` instead of a Bunch
        object.

    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes:

        data : ndarray, shape (101766, 24)
            Each row corresponding to the 24 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (101766,)
            Each value represents whether readmission of the patient
            occurred within 30 days of the release.
        feature_names : list of length 24
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the Diabetes 130-Hospitals dataset.
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
    .. footbibliography::

    """
    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    # For data_home see
    # https://github.com/scikit-learn/scikit-learn/issues/27447
    return fetch_openml(
        data_id=43874,
        data_home=str(data_home),
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
        parser="auto",
    )
