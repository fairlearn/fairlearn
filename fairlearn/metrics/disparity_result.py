# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


class DisparityResult:
    def __init__(self):
        # The 'disparity' property is computed
        self.disparity = None
        # The 'group_metric' dictionary comes from the given metric_function
        self.group_metric = {}
