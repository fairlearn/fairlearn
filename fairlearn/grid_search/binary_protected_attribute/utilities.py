# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def generate_protected_attribute_info(protected_attribute):
    unique_labels, counts = np.unique(
        protected_attribute, return_counts=True)
    if len(unique_labels) > 2:
        raise RuntimeError("Protected Attribute is not binary")

    p0 = counts[0] / len(protected_attribute)
    # It is possible that our data only contain one value for
    # the protected attribute
    # We do not consider this an error
    if len(counts) > 1:
        p1 = counts[1] / len(protected_attribute)
    else:
        p1 = 0

    return p0, p1, unique_labels[0]
