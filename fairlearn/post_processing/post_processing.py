# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class PostProcessing:
    def fit(self, X, y, protected_attribute):
        raise NotImplementedError(self.fit.__name__ + " is not implemented")

    def predict(self, X, protected_attribute):
        raise NotImplementedError(self.predict.__name__ + " is not implemented")

    def predict_proba(self, X, protected_attribute):
        raise NotImplementedError(self.predict_proba.__name__ + " is not implemented")
