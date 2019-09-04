# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


class DemographicParity:
    def __init__(self):
        # Nothing to do (yet)
        pass

    def set_binary_protected_attribute(self, protected_attribute):
        self.protected_attribute = protected_attribute

        unique_labels, counts = np.unique(protected_attribute, return_counts=True)
        if len(unique_labels) > 2:
            raise RuntimeError("Protected Attribute contains more than two unique values")

        self.p0 = counts[0] / len(protected_attribute)
        self.p1 = 1 - self.p0
        self.a0_val = unique_labels[0]

    def generate_binary_trade_offs(self, number_of_trade_offs):
        limit = 1
        if self.p1 > 0 and self.p0/self.p1 > 1:
            limit = self.irness_metric.p0/self.p1
        return np.linspace(-2 * limit, 2 * limit, number_of_trade_offs)

    def reweight_and_relabel_binary_attribute(self, Y, trade_off):
        weight_func = np.vectorize(self._weight_function)
        weights = weight_func(Y, self.protected_attribute, trade_off, self.p1/self.p0, self.a0_val)

        def f(x): return 1 if x > 0 else 0
        re_labels = np.vectorize(f)(weights)

        return weights, re_labels

    def _weight_function(self, y_val, a_val, L, p_ratio, a0_val):
        if a_val == a0_val:
            return 2 * y_val - 1 - L * p_ratio
        else:
            return 2 * y_val - 1 + L
