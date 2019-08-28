# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def generate_protected_attribute_info(protected_attribute):
    unique_labels, counts = np.unique(
        protected_attribute, return_counts=True)
    if len(unique_labels) > 2:
        raise RuntimeError("Protected Attribute contains "
                           "more than two unique values")

    p0 = counts[0] / len(protected_attribute)
    p1 = 1 - p0

    return p0, p1, unique_labels[0]
