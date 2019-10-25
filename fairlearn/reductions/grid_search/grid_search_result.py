# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GridSearchResult:
    """Class to hold a single result from the :class:`GridSearch` class.
    """

    def __init__(self, predictor, lambda_vec, objective, gamma):
        self._predictor = predictor
        self._lambda_vec = lambda_vec
        self._objective = objective
        self._gamma = gamma

    @property
    def predictor(self):
        """The predictor trained for this particular result
        from :class:`GridSearch`
        """
        return self._predictor

    @property
    def lambda_vec(self):
        """The Lagrange multiplier corresponding to this
        result. The exact contents of this are defined
        by the `constraints` argument passed to
        :class:`GridSearch`
        """
        return self._lambda_vec

    @property
    def objective(self):
        """Description goes here
        """
        return self._objective

    @property
    def gamma(self):
        """Description goes here
        """
        return self._gamma
