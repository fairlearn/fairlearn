# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from ._constants import _DOWNLOAD_DIRECTORY_NAME


def fetch_acs_income(
    *,
    cache=True,
    data_home=None,
    as_frame=True,
    return_X_y=False,
    states=None,
):
    """Load the ACS Income dataset (regression).

    Download it if necessary.

    ==============   ====================
    Samples total                 1664500
    Dimensionality                     10
    Features         numeric, categorical
    Target                        numeric
    ==============   ====================

    Source:

    - Paper: Ding et al. (2021) :footcite:`ding2021retiring`
    - Repository: https://github.com/zykls/folktables/

    Read more in the :ref:`User Guide <acsincome_data>`.

    .. versionadded:: 0.8.0

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

    states: list, default=None
        List containing two letter (capitalized) state abbreviations.
        If None, data from all 50 US states and Puerto Rico will be returned.
        Note that Puerto Rico is the only US territory included in this dataset.
        The state abbreviations and codes can be found on page 1 of the data
        dictionary at ACS PUMS :footcite:`census2019pums`.

    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (1664500, 10)
            Each row corresponding to the 10 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (1664500,)
            Integer denoting each person's annual income.
            A threshold can be applied as a postprocessing step to frame this
            as a binary classification problem.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 10
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the ACSIncome dataset.
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
    # State Code based on 2010 Census definitions
    _STATE_CODES = {
        "AL": "01",
        "AK": "02",
        "AZ": "04",
        "AR": "05",
        "CA": "06",
        "CO": "08",
        "CT": "09",
        "DE": "10",
        "FL": "12",
        "GA": "13",
        "HI": "15",
        "ID": "16",
        "IL": "17",
        "IN": "18",
        "IA": "19",
        "KS": "20",
        "KY": "21",
        "LA": "22",
        "ME": "23",
        "MD": "24",
        "MA": "25",
        "MI": "26",
        "MN": "27",
        "MS": "28",
        "MO": "29",
        "MT": "30",
        "NE": "31",
        "NV": "32",
        "NH": "33",
        "NJ": "34",
        "NM": "35",
        "NY": "36",
        "NC": "37",
        "ND": "38",
        "OH": "39",
        "OK": "40",
        "OR": "41",
        "PA": "42",
        "RI": "44",
        "SC": "45",
        "SD": "46",
        "TN": "47",
        "TX": "48",
        "UT": "49",
        "VT": "50",
        "VA": "51",
        "WA": "53",
        "WV": "54",
        "WI": "55",
        "WY": "56",
        "PR": "72",
    }
    # number of features
    _NUM_FEATS = 10

    # check that user-provided state abbreviations are valid
    if states is not None:
        states = [state.upper() for state in states]
        for state in states:
            try:
                _STATE_CODES[state]
            except KeyError:
                raise KeyError(
                    f"Error with state code: {state}\n"
                    "State code must be a two letter abbreviation"
                    f"from the list {list(_STATE_CODES.keys())}\n"
                    "Note that PR is the abbreviation for Puerto Rico."
                )
    else:
        states = _STATE_CODES.keys()

    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    # fetch data for all 50 US states and Puerto Rico
    # For data_home see
    # https://github.com/scikit-learn/scikit-learn/issues/27447
    data_dict = fetch_openml(
        data_id=43141,
        data_home=str(data_home),
        cache=cache,
        as_frame=True,
        return_X_y=False,
        parser="auto",
    )

    # filter by state
    df_all = data_dict["data"].copy(deep=True)
    df_all["PINCP"] = data_dict["target"]
    cols = df_all.columns
    df = pd.DataFrame(np.zeros((0, len(cols))), columns=cols)
    for state in states:
        dfs = [df, df_all.query(f"ST == {int(_STATE_CODES[state])}")]
        df = pd.concat(dfs)
    # drop the state column since it is not a feature in the published ACSIncome dataset
    df.drop("ST", axis=1, inplace=True)

    if as_frame:
        data_dict["data"] = df.iloc[:, :_NUM_FEATS]
        data_dict["frame"] = df
        data_dict["target"] = df.iloc[:, _NUM_FEATS]
    else:
        data_dict["data"] = df.iloc[:, :_NUM_FEATS].values
        data_dict["frame"] = None
        data_dict["target"] = df.iloc[:, _NUM_FEATS].values

    output = data_dict

    if return_X_y:
        output = (data_dict["data"], data_dict["target"])

    return output
