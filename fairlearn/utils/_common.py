# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.


import inspect


def _get_soft_predictions(estimator, X, predict_method):
    r"""Return soft predictions of a classifier.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn compatible estimator

    X : array-like
        The input for which the output is desired

    predict_method : {'auto', 'predict_proba', 'decision_function', 'predict'\
            }, default='auto'

        Defines which method of the ``estimator`` is used to get the output
        values.

        - 'auto': use one of ``predict_proba``, ``decision_function``, or
          ``predict``, in that order.
        - 'predict_proba': use the second column from the output of
          `predict_proba`. It is assumed that the second column represents the
          positive outcome.
        - 'decision_function': use the raw values given by the
          `decision_function`.
        - 'predict': use the hard values reported by the `predict` method if
          estimator is a classifier, and the regression values if estimator is
          a regressor. This is equivalent to what is done in
          :footcite:`hardt2016equality`.

    Returns
    -------
    predictions : ndarray
        The output from estimator's desired predict method.
    """
    if predict_method == "auto":
        if hasattr(estimator, "predict_proba"):
            predict_method = "predict_proba"
        elif hasattr(estimator, "decision_function"):
            predict_method = "decision_function"
        else:
            predict_method = "predict"

    output = getattr(estimator, predict_method)(X)
    if predict_method == "predict_proba":
        return output[:, 1]
    return output


def _filter_kwargs(func, kwargs):
    """Helper function to filter kwargs that are accepted by `func` according to its
    signature. Returns all `kwargs` if `func` accepts `**kwargs` (or similar).

    Parameters
    ----------
    func : callable
        Function or method to pass params to.
    kwargs : dict
        Dictionary of params to pass to `func`.

    Returns
    -------
    filtered_kwargs : dict
    """
    sig = inspect.signature(func)

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    if accepts_var_kwargs:
        return kwargs

    else:
        valid_params = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in valid_params}
