from sklearn.utils.validation import check_array as _check_array


def validate_data(estimator, X, **kwargs):
    """
    Validate data for sklearn estimators. This function provides compatibility across
    different sklearn versions by handling the validation method changes between versions.

    Parameters
    ----------
    estimator : object
        Sklearn estimator instance
    X : array-like
        Input data to validate
    **kwargs : dict
        Additional keyword arguments passed to the validation function

    Returns
    -------
    array-like
        The validated input data

    Notes
    -----
    For sklearn versions >= 1.6, uses estimator._validate_data()
    For sklearn versions < 1.6, uses sklearn.utils.validation.validate_data()
    """
    try:
        from sklearn.utils.validation import validate_data

        return validate_data(estimator, X, **kwargs)
    except ImportError:
        # sklearn version < 1.6
        return estimator._validate_data(X, **kwargs)


def check_array(X, **kwargs):
    """
    Wrapper for sklearn's check_array function that handles version compatibility.
    Using this private function is only necessary if the call contains
    force_all_finite for sklearn version < 1.6 or ensure_all_finite for sklearn version >= 1.6.
    Parameters
    ----------
    X : array-like
        Input array to check/convert
    **kwargs : dict
        Additional arguments to pass to sklearn's check_array function

    Returns
    -------
    array-like
        The checked and potentially converted input array

    Notes
    -----
    This function provides compatibility across different sklearn versions by
    handling changes in the check_array function behavior.
    """
    try:
        return _check_array(X, **kwargs)
    except TypeError:
        # sklearn version < 1.6
        kwargs.pop("ensure_all_finite", False)
        return _check_array(X, **kwargs, force_all_finite=False)
