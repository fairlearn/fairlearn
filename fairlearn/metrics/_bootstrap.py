# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility functions for bootstrap calculations (used for confidence intervals)."""

def process_ci_bounds(ci):
    '''Interpret user inputs to confidence interval function.'''

    if isinstance(ci, (int, float)) and (0 <= ci <= 1):
        return (0.5 - ci/2, 0.5 + ci/2)
    elif isinstance(ci, (tuple, list)):
        if all(((0 <= x <= 1) for x in ci)):
            return tuple(ci)
        
    raise ValueError(f"All desired confidence interval bounds must be within [0, 1].")