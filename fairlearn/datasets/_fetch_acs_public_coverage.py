# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pathlib

from sklearn.datasets import fetch_openml
from ._constants import _DOWNLOAD_DIRECTORY_NAME

# State Code based on 2010 Census definitions
_STATE_CODES = {'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
                'CO': '08', 'CT': '09', 'DE': '10', 'FL': '12', 'GA': '13',
                'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18', 'IA': '19',
                'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23', 'MD': '24',
                'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28', 'MO': '29',
                'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33', 'NJ': '34',
                'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38', 'OH': '39',
                'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44', 'SC': '45',
                'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49', 'VT': '50',
                'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55', 'WY': '56',
                'PR': '72'}


class InvalidStateException(Exception):
    pass


def check_states_valid(states):
    for st in states:
        if st not in _STATE_CODES:
            raise InvalidStateException(f"'{st}' is an invalid state code.\n"
                                        f"State code must be a two letter abbreviation"
                                        f"from the list {list(_STATE_CODES.keys())}.\n"
                                        f"Note that PR is the abbreviation for Puerto Rico.")


def fetch_acs_public_coverage(*, cache=True, data_home=None,
                            as_frame=False, return_X_y=False,
                            states=None
                            ):
    """Load the ACS Public Coverage dataset.

    Download it if necessary.

    ==============   ==============
    Samples total         1,138,289
    Dimensionality               19
    Features                   real
    Target                     real
    ==============   ==============

    Source: Paper: Frances Ding et al. (2021) [1]_
            and corresponding repository https://github.com/zykls/folktables/

    .. versionadded:: 0.7.1

    Prediction task is to determine whether a person has public
    health coverage.

    Parameters
    ----------
    cache : bool, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : str, default=None
        Specify another download and cache folder for the datasets.
        By default, all fairlearn data is stored in '~/.fairlearn-data'
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

    states : list, default=None
        List containing two letter (capitalized) state abbreviations.
        If None, data from all 50 US states and Puerto Rico will be returned.
        Note that Puerto Rico is the only US territory included in this dataset.
        The state abbreviations and codes can be found on page 1 of the data
        dictionary at ACS PUMS:
        https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2018.pdf

    Returns
    -------
    dataset : :obj:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (1138289, 20)
            Each row corresponding to the 19 feature values in order as well as
            the target variable (PUBCOV).
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (1138289,)
            Each value represents whether the person has public health coverage.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 20
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the ACS Public Coverage dataset.

    (data, target) : tuple of (numpy.ndarray, numpy.ndarray)
        if ``return_X_y`` is True and ``as_frame`` is False

    (data, target) : tuple of (pandas.DataFrame, pandas.Series)
        if ``return_X_y`` is True and ``as_frame`` is True

    References
    ----------
    .. [1] Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021).
       "Retiring Adult: New Datasets for Fair Machine Learning."
       Advances in Neural Information Processing Systems, 34.

    """

    if not data_home:
        data_home = pathlib.Path().home() / _DOWNLOAD_DIRECTORY_NAME

    data = fetch_openml(
        data_id=43140,
        data_home=data_home,
        cache=cache,
        as_frame=True,
        return_X_y=False,
    )

    if states:
        states = [st.upper() for st in states]
        check_states_valid(states)

        query = " or ".join(f"ST == {float(_STATE_CODES[st])}" for st in states)
        indices = data.data.query(query).index.to_list()

        data.data = data.data.query(query)
        data.target = data.target.iloc[indices]
        data.frame = data.frame.query(query)

    if not as_frame:
        data.data = data.data.to_numpy()
        data.target = data.target.to_numpy()
        data.frame = None

    if return_X_y:
        return (data.data, data.target)

    return data