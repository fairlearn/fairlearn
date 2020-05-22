import warnings
from ._common import FairnessWarning

from sklearn.datasets import fetch_openml


def fetch_boston(
    *,
    version='1',
    cache=True,
    data_home=None,
    as_frame=False,
    return_X_y=False,
    warn=True,
):
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
    that live in a neighborhood (B). This dataset should be used to explore
    and benchmark fairness. Predicting a house-price while ignoring these
    sensitive attributes is naive.

    Parameters
    ----------
    version : integer or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.

    cache : boolean, default=True
        Whether to cache downloaded datasets using joblib.

    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

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

    warn : boolean, default=False.
        If True, it raises an extra warning to make users aware of the unfairness
        aspect of this dataset.


    Returns
    -------
    dataset : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : ndarray, shape (506, 14)
            Each row corresponding to the 8 feature values in order.
            If ``as_frame`` is True, ``data`` is a pandas object.
        target : numpy array of shape (20640,)
            Each value corresponds to the average
            house value in units of 100,000.
            If ``as_frame`` is True, ``target`` is a pandas object.
        feature_names : list of length 8
            Array of ordered feature names used in the dataset.
        DESCR : string
            Description of the California housing dataset.

    Notes
    -----
    This dataset consists of 506 samples and 14 features.
    """
    if warn:
        msg = "You are about to use with an unfair dataset. Mind the `B` and `LSTAT` columns."
        warnings.warn(FairnessWarning(msg))
    return fetch_openml(
        data_id=531,
        data_home=data_home,
        version=version,
        cache=cache,
        as_frame=as_frame,
        return_X_y=return_X_y,
    )
