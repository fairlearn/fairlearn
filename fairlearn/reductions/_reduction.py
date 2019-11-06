# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class Reduction:
    """Base class for our reduction-implementing estimators
    """

    def fit(self, X, y, **kwargs):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()


# Ensure that Reduction shows up in correct place in documentation
# when it is used as a base class
Reduction.__module__ = "fairlearn.reductions"
