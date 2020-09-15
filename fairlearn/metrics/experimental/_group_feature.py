# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder


class GroupFeature:
    def __init__(self, base_name, feature_vector, index, name):
        self._encoder = LabelEncoder()
        self._encoded = np.asarray(self._encoder.fit_transform(feature_vector))

        self._name = "{0} {1}".format(base_name, index)
        if name is not None:
            self._name = name
        elif isinstance(feature_vector, pd.Series):
            if feature_vector.name is not None:
                self._name = feature_vector.name

    def get_mask_for_class(self, target_class):
        idx = self.classes.index(target_class)
        return (self._encoded == idx)

    @property
    def name(self):
        return self._name

    @property
    def classes(self):
        return list(self._encoder.classes_)
