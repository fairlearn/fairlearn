# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class ReductionsLearner:
    def fit(self, X, y, **kwargs):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()

    def posterior_predict(self, X):
        raise NotImplementedError()

    def posterior_predict_proba(self, X):
        raise NotImplementedError()
