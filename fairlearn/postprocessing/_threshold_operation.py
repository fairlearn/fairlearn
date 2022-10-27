# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


class ThresholdOperation:
    """Represents the threshold operations that are used in postprocessing approaches.

    Threshold operations simply indicate a threshold and an operator, thereby defining a function.
    The function can be evaluated at arbitrary points (usually the scores returned from
    unconstrained predictors) to return a bool value.

    Read more in the :ref:`User Guide <postprocessing>`.

    :param operator: the threshold operator, can be either '>' or '<'
    :type operator: str
    :param threshold: the threshold, can be numpy.inf or -numpy.inf
    :type threshold: float
    """

    def __init__(self, operator, threshold):
        if operator not in [">", "<"]:
            raise ValueError("Unrecognized operator: " + operator)
        self._operator = operator
        self._threshold = threshold

    @property
    def threshold(self):
        """Return the stored threshold."""
        return self._threshold

    @property
    def operator(self):
        """Return the stored threshold operator."""
        return self._operator

    def __call__(self, y_hat):
        """Evaluate the threshold rule `y_hat > t` or `y_hat < t`.

        :param y_hat: the input array
        :type y_hat: array
        :return: the result of elementwise application of the threshold rule
        :rtype: array
        """
        if self._operator == ">":
            return y_hat > self._threshold
        elif self._operator == "<":
            return y_hat < self._threshold
        else:
            raise ValueError("Unrecognized operator: " + self._operator)

    def __repr__(self):  # noqa: D105
        return "[{}{}]".format(self._operator, self._threshold)
