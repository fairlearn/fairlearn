# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import pandas as pd

import fairlearn.moments as moments


class SimpleQualityMetric:
    def __init__(self):
        self.error_metric = moments.MisclassificationError()
        self.disparity_metric = moments.DP()

    def set_data(self, X, Y, protected_attribute):
        self.X = X
        self.Y = Y
        self.protected_attribute = protected_attribute

        self.error_metric.init(X, protected_attribute, pd.Series(Y))
        self.disparity_metric.init(X, protected_attribute, pd.Series(Y))

    def get_quality(self, model):
        current_error_metric = copy.deepcopy(self.error_metric)
        current_disparity_metric = copy.deepcopy(self.disparity_metric)

        def classifier(X): return model.predict(X)
        current_error = current_error_metric.gamma(classifier)[0]
        current_disparity = current_disparity_metric.gamma(classifier).max()

        return -(current_error+current_disparity)
