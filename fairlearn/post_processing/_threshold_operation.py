# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class ThresholdOperation():
    def __init__(self, operator, threshold):
        if operator not in ['>', '<']:
            raise ValueError("Unrecognized operator: " + operator)
        self._operator = operator
        self._threshold = threshold

    @property
    def threshold(self):
        return self._threshold

    @property
    def operator(self):
        return self._operator

    def get_predictor_from_operation(self):
        """ Encodes the threshold rule Y_hat > t or Y_hat < t"""
        if self._operator == '>':
            return lambda x: x > self._threshold
        elif self._operator == '<':
            return lambda x: x < self._threshold
        else:
            raise ValueError("Unrecognized operator: " + self._operator)

    def __repr__(self):
        return "[{}{}]".format(self._operator, self._threshold)
