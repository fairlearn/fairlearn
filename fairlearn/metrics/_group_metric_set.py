# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import fairlearn.metrics as fm
import fairlearn.metrics._input_manipulations as fmim


class GroupMetricSet:
    """Class to hold a collection of GroupMetricResult objects."""

    BINARY_CLASSIFICATION = 'binary_classification'

    REGRESSION = 'regression'

    def __init__(self):
        self._model_type = None
        self._y_true = None
        self._y_pred = None
        self._groups = None
        self._group_names = None
        self._metrics = None

    @property
    def model_type(self):
        """Return the type of model being described."""
        return self._model_type

    @model_type.setter
    def model_type(self, value):
        self._model_type = value

    @property
    def y_true(self):
        """Return the array of y_true values."""
        return self._y_true

    @y_true.setter
    def y_true(self, value):
        self._y_true = fmim._convert_to_ndarray_1d(value)

    @property
    def y_pred(self):
        """Return the array of y_pred values."""
        return self._y_pred

    @y_pred.setter
    def y_pred(self, value):
        self._y_pred = fmim._convert_to_ndarray_1d(value)

    @property
    def groups(self):
        """Return the array of group values."""
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = fmim._convert_to_ndarray_1d(value)

    @property
    def group_names(self):
        """Return the group_names dictionary."""
        return self._group_names

    @group_names.setter
    def group_names(self, value):
        key_types = set(type(k) for k in value.keys())
        if key_types != {int}:
            raise ValueError("Keys not integers")
        value_types = set(type(v) for v in value.values())
        if value_types != {str}:
            raise ValueError("Values not strings")

        self._group_names = value

    @property
    def metrics(self):
        """Return the GUID-GroupMetricResult dictionary of group metrics."""
        return self._metrics

    @metrics.setter
    def metrics(self, value):
        key_types = set(type(k) for k in value.keys())
        if key_types != {str}:
            raise ValueError("Keys not strings")
        value_types = set(type(v) for v in value.values())
        if value_types != {fm.GroupMetricResult}:
            raise ValueError("Values not GroupMetricResults")
        self._metrics = value
