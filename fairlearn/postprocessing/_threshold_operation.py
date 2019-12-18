# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class ThresholdOperation():
    """Represents the threshold operations that are used in postprocessing approaches.

    Threshold operations simply indicate a threshold and an operator, thereby defining a function.
    The function can be evaluated at arbitrary points (usually the scores returned from
    unconstrained predictors) to return a bool value.

    :param operator: the threshold operator, can be either '>' or '<'
    :type operator: str
    :param threshold: the threshold, can be numpy.inf or -numpy.inf
    :type threshold: float
    """

    def __init__(self, operator, threshold):
        if operator not in ['>', '<']:
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

    def get_predictor_from_operation(self):
        """Encode the threshold rule `Y_hat > t` or `Y_hat < t`.

        :return: a function that takes a single argument to evaluate it against the threshold rule
        :rtype: lambda
        """
        if self._operator == '>':
            return lambda x: x > self._threshold
        elif self._operator == '<':
            return lambda x: x < self._threshold
        else:
            raise ValueError("Unrecognized operator: " + self._operator)

    def __repr__(self):  # noqa: D105
        return "[{}{}]".format(self._operator, self._threshold)
