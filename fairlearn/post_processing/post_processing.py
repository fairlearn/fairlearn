# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class PostProcessing:
    def fit(self, X, y, aux_data, **kwargs):
        raise NotImplementedError(self.fit.__name__ + " is not implemented")

    def predict(self, X, aux_data):
        raise NotImplementedError(self.predict.__name__ + " is not implemented")

    def predict_proba(self, X, aux_data):
        raise NotImplementedError(self.predict_proba.__name__ + " is not implemented")
