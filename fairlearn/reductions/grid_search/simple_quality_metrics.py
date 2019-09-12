# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import pandas as pd

import fairlearn.moments as moments


class SimpleClassificationQualityMetric:
    """Class to calculate a metric for comparing models produced
    by GridSearch
    The metric produced is simply the sum of error and disparity
    for the given model
    """

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


class SimpleRegressionQualityMetric:
    """Simple class to produce a quality metric for regression models
    produced by GridSearch, to enable one to be selected for the
    predict methods
    """

    def set_data(self, X, Y, protected_attribute):
        self.X = X
        self.Y = Y
        self.protected_attribute = protected_attribute

    def get_quality(self, model):
        labels = pd.Series(self.Y)
        preds = pd.Series(model.predict(self.X))
        attrs = pd.Series(self.protected_attribute)
        attr_vals = attrs.unique()
        errors = (preds-labels)**2
        error = errors.mean()
        error0 = errors[attrs == attr_vals[0]].mean()
        error1 = errors[attrs == attr_vals[1]].mean()
        return -(error+error0+error1)
