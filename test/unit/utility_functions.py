# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np


def logging_all_close(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Wrap numpy.all_close functionality with print() on failure.

    Function arguments match numpy.all_close (assuming that it uses
    numpy.isclose), but if there are failures, then some diagnostics
    will be printed to stdout.
    """
    aa = np.asarray(a)
    ba = np.asarray(b)

    match_mask = np.isclose(aa, ba, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if not np.all(match_mask):
        print("a mismatches: ", aa[np.logical_not(match_mask)])
        print("b mismatches: ", ba[np.logical_not(match_mask)])
        print("mismatch indices: ", np.where(np.logical_not(match_mask)))

    return np.all(match_mask)
