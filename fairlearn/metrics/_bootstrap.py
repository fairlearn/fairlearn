# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility functions for bootstrap calculations (used for confidence intervals)."""

from warnings import warn
import pandas as pd
import numpy as np
from scipy.stats import norm

def process_ci_bounds(ci, precision=6):
    '''Interpret user inputs to confidence interval function.'''

    if isinstance(ci, (int, float)) and (0 <= ci <= 1):
        return tuple(np.round((0.5 - ci/2, 0.5 + ci/2), decimals=precision))
    elif isinstance(ci, (tuple, list)):
        if all(((0 <= x <= 1) for x in ci)):
            return tuple(ci)
        
    raise ValueError(f"All desired confidence interval bounds must be within [0, 1].")

def create_ci_output(bootstrap_runs, ci, sample_estimate):
    '''Format confidence interval output to look like prototype object.'''
    bootstrap_quantiles = np.quantile(bootstrap_runs, q=ci, axis=0)
    prototype_index = sample_estimate.index

    bootstrap_mean = np.mean(bootstrap_runs, axis=0)
    bootstrap_std = np.std(bootstrap_runs, axis=0)
    if isinstance(sample_estimate, pd.Series):
        quantiles = [
            (quantile, pd.Series(qmf, index=prototype_index))
            for quantile, qmf in zip(ci, bootstrap_quantiles)
        ]
        bootstrap_mean = pd.Series(bootstrap_mean, index=prototype_index)
        bootstrap_std = pd.Series(bootstrap_std, index=prototype_index)

    else:
        prototype_columns = sample_estimate.columns
        quantiles = [
            (quantile, pd.DataFrame(qmf, index=prototype_index, columns=prototype_columns))
            for quantile, qmf in zip(ci, bootstrap_quantiles)
        ]
        bootstrap_mean = pd.DataFrame(bootstrap_mean, index=prototype_index, columns=prototype_columns)
        bootstrap_std = pd.DataFrame(bootstrap_std, index=prototype_index, columns=prototype_columns)

    # Calculate bias
    bias = bootstrap_mean - sample_estimate
    bias_adjusted_mean_estimate = 2*sample_estimate - bootstrap_mean

    # Efron and Tibshirani (1993) guidance for warnings on bootstrap bias
    high_bias_regions = bias >= 0.25 * bootstrap_std
    if np.any(high_bias_regions):
        warn(
            f"""Some statistics calculated by MetricFrame are detected to have high bias 
            and might be a skewed representation of the overall population."""
        )
        print(high_bias_regions) # TODO: Handle this better

    return quantiles