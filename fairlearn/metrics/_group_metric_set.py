# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class GroupMetricSet:
    """Class to hold a collection of GroupMetricResult objects."""

    BINARY_CLASSIFICATION = 'binary_classification'

    REGRESSION = 'regression'

    @property
    def model_type(self):
        """Return the type of model being described."""
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = value
