# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class QualityMetric:
    def set_data(self, X, Y, bin_id):
        raise NotImplementedError()

    def get_quality(self, model):
        raise NotImplementedError()
