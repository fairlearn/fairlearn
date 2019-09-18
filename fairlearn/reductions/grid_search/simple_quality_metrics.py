# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import pandas as pd

from fairlearn.reductions.grid_search import QualityMetric

import fairlearn.moments as moments


class SimpleClassificationQualityMetric(QualityMetric):
    """Class to calculate a metric for comparing models produced
    by GridSearch
    The metric produced is simply the sum of error and disparity
    for the given model
    """

    def __init__(self):
        self.error_metric = moments.MisclassificationError()
        self.disparity_metric = moments.DP()

    def set_data(self, X, Y, bin_id):
        self.X = X
        self.Y = Y
        self.bin_id = bin_id

        y_series = None
        if isinstance(self.Y, pd.DataFrame):
            y_series = self.Y[0]
        else:
            if len(self.Y.shape) == 1:
                y_series = pd.Series(self.Y)
            elif len(self.Y.shape) == 2 and self.Y.shape[1] == 1:
                y_series = pd.Series(self.Y[:, 0])
            else:
                raise RuntimeError("SimpleClassificationQualityMetric got bad Y")

        self.error_metric.init(X, bin_id, y_series)
        self.disparity_metric.init(X, bin_id, y_series)

    def get_quality(self, model):
        current_error_metric = copy.deepcopy(self.error_metric)
        current_disparity_metric = copy.deepcopy(self.disparity_metric)

        def classifier(X): return model.predict(X)
        current_error = current_error_metric.gamma(classifier)[0]
        current_disparity = current_disparity_metric.gamma(classifier).max()

        return -(current_error+current_disparity)


class SimpleRegressionQualityMetric(QualityMetric):
    """Simple class to produce a quality metric for regression models
    produced by GridSearch, to enable one to be selected for the
    predict methods
    """

    def set_data(self, X, Y, bin_id):
        self.X = X
        self.Y = Y
        self.bin_id = bin_id

    def get_quality(self, model):
        labels = pd.Series(self.Y)
        preds = pd.Series(model.predict(self.X))
        attrs = pd.Series(self.bin_id)
        attr_vals = attrs.unique()
        errors = (preds-labels)**2
        error = errors.mean()
        error0 = errors[attrs == attr_vals[0]].mean()
        error1 = errors[attrs == attr_vals[1]].mean()
        return -(error+max(error0, error1))
