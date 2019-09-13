# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class PostProcessing:
    def fit(self, X, y, protected_attribute):
        raise NotImplementedError()

    def predict(self, X, protected_attribute):
        raise NotImplementedError()

    def predict_proba(self, X, protected_attribute):
        raise NotImplementedError()
