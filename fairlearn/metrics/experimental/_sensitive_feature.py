# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


class SensitiveFeature:
    def __init__(self, feature_vector, index):
        self._encoder = LabelEncoder()
        self._encoded = np.asarray(self._encoder.fit_transform(feature_vector))

        feature_vector = "SF {0}".format(index)
        if isinstance(feature_vector, pd.Series):
            if feature_vector.name is not None:
                self._name = feature_vector.name

    def get_mask(self, class_index):
        assert class_index is int
        return (self._encoded == class_index)

    @property
    def name(self):
        return self._name

    @property
    def classes(self):
        return list(self._encoder.classes_)
