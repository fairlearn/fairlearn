# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from test.unit.input_convertors import ensure_list, ensure_series


def is_invalid_transformation(**kwargs):
    A_two_dim = kwargs["A_two_dim"]
    transform = kwargs["transformA"]
    if A_two_dim and transform in [ensure_list, ensure_series]:
        return True
    return False
