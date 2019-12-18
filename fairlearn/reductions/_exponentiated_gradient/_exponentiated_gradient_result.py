# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


# Class is just for holding results so 'imperative mode' makes
# little sense
# flake8: noqa: D401

class ExponentiatedGradientResult:
    """Class to hold the result of an `ExponentiatedGradient` estimator."""

    def __init__(self, best_classifier, best_gap, classifiers, weights, last_t, best_t,
                 n_oracle_calls):
        """Result object for the exponentiated gradient reduction operation."""
        self._best_classifier = best_classifier
        self._best_gap = best_gap
        self._classifiers = classifiers
        self._weights = weights
        self._last_t = last_t
        self._best_t = best_t
        self._n_oracle_calls = n_oracle_calls

    @property
    def best_classifier(self):
        """The best classifier found by the algorithm.

        A function that maps a DataFrame `X` containing covariates to a Series containing the
        corresponding probabilistic decisions in :math:`[0,1]`
        """
        return self._best_classifier

    @property
    def best_gap(self):
        """The quality of `best_classifier`.

        If the algorithm has converged then :code:`best_gap <= nu`;
        the solution `best_classifier` is guaranteed to have the classification error within
        :code:`2*best_gap` of the best error under constraint `eps`; the constraint violation
        is at most :code:`2*(eps+best_gap)`
        """
        return self._best_gap

    @property
    def classifiers(self):
        """The base classifiers generated (instances of estimator)."""
        return self._classifiers

    @property
    def weights(self):
        """The weights of those classifiers within `best_classifier`."""
        return self._weights

    @property
    def last_t(self):
        """The last executed iteration; always :code:`last_t < T`."""
        return self._last_t

    @property
    def best_t(self):
        """The iteration in which best_classifier was obtained."""
        return self._best_t

    @property
    def n_oracle_calls(self):
        """The number of times the estimator was called."""
        return self._n_oracle_calls

    def _as_dict(self):
        return {
            "best_classifier": self._best_classifier,
            "best_gap": self._best_gap,
            "classifiers": self._classifiers,
            "weights": self._weights,
            "last_t": self._last_t,
            "best_t": self._best_t,
            "n_oracle_calls": self._n_oracle_calls
        }
