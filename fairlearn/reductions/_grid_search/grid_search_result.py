# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GridSearchResult:
    """Class to hold a single result from the :class:`GridSearch` class."""

    def __init__(self, predictor, lambda_vec, objective, gamma, oracle_call_execution_time):
        self._predictor = predictor
        self._lambda_vec = lambda_vec
        self._objective = objective
        self._gamma = gamma
        self._oracle_call_execution_time = oracle_call_execution_time

    @property
    def predictor(self):
        """Return the predictor trained for this particular result from :class:`GridSearch`."""
        return self._predictor

    @property
    def lambda_vec(self):
        """Return the Lagrange multiplier corresponding to this result.

        The exact contents of this are defined by the `constraints`
        argument passed to :class:`GridSearch`
        """
        return self._lambda_vec

    @property
    def objective(self):
        """Description goes here."""  # noqa: D401
        return self._objective

    @property
    def gamma(self):
        """Description goes here."""  # noqa: D401
        return self._gamma

    @property
    def oracle_call_execution_time(self):
        """Measures the time it took to make the oracle call.

        :return: the execution time of the oracle call
        :rtype: float
        """
        return self._oracle_call_execution_time
