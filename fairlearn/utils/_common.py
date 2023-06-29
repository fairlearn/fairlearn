# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from __future__ import annotations

_MESSAGE_BAD_COSTS = (
    "costs needs to be a dictionary with keys "
    "'fp' and 'fn' containing non-negative values, which are not both zero"
)


def unpack_fp_fn_costs(costs: dict) -> tuple[float, float]:
    """Validates and unpacks the given `costs`.

    Parameters
    ----------
    costs : dict
        A dictionary detailing the cost for false positives and false negatives,
        of the form :code:`{'fp': <fp_cost>, 'fn': <fn_cost>}`.
        
    Returns
    -------
    tuple[float, float]
        A tuple respectively composed of the cost of false positives and the
        cost of false negatives, i.e., a tuple with 
        :code:`(fp_cost, fn_cost)`.

    Raises
    ------
    ValueError
        Raised when the provided costs are invalid (e.g., missing keys
        in the provided dict, or negative costs).
    """
    if (
        type(costs) is dict
        and costs.keys() == {"fp", "fn"}
        and costs["fp"] >= 0.0
        and costs["fn"] >= 0.0
        and costs["fp"] + costs["fn"] > 0.0
    ):
        fp_cost = costs["fp"]
        fn_cost = costs["fn"]

    else:
        raise ValueError(_MESSAGE_BAD_COSTS)
    
    return fp_cost, fn_cost


def _get_soft_predictions(estimator, X, predict_method):
    r"""Return soft predictions of a classifier.

    Parameters
    ----------
    estimator : estimator
        A scikit-learn compatible estimator

    X : array-like
        The intput for which the output is desired

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
          a regressor. This is equivalent to what is done in [1]_.

    Returns
    -------
    predictions : ndarray
        The output from estimator's desired predict method.

    References
    ----------
    .. [1] M. Hardt, E. Price, and N. Srebro, "Equality of Opportunity in
       Supervised Learning," arXiv.org, 07-Oct-2016.
       [Online]. Available: https://arxiv.org/abs/1610.02413.
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
