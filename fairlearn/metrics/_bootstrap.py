# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility functions for bootstrap calculations (used for confidence intervals)."""

import pandas as pd
import numpy as np

def process_ci_bounds(ci, precision=6):
    '''Interpret user inputs to confidence interval function.'''

    if isinstance(ci, (int, float)) and (0 <= ci <= 1):
        return tuple(np.round((0.5 - ci/2, 0.5 + ci/2), decimals=precision))
    elif isinstance(ci, (tuple, list)):
        if all(((0 <= x <= 1) for x in ci)):
            return tuple(ci)
        
    raise ValueError(f"All desired confidence interval bounds must be within [0, 1].")

def create_ci_output(bootstrap_runs, ci, prototype):
    '''Format confidence interval output to look like prototype object.'''
    bootstrap_quantiles = np.quantile(bootstrap_runs, q=ci, axis=0)
    prototype_index = prototype.index
    if isinstance(prototype, pd.Series):
        return [
            {quantile : pd.Series(qmf, index=prototype_index)}
            for quantile, qmf in zip(ci, bootstrap_quantiles)
        ]
    else:
        prototype_columns = prototype.columns
        return [
            {quantile : pd.DataFrame(qmf, index=prototype_index, columns=prototype_columns)}
            for quantile, qmf in zip(ci, bootstrap_quantiles)
        ]