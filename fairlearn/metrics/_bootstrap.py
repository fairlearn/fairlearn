# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility functions for bootstrap calculations (used for confidence intervals)."""

from warnings import warn
import pandas as pd
import numpy as np
from scipy.stats import norm
import numbers


def process_ci_bounds(ci, ci_method, precision=6):
    """Parses user inputs to construct confidence intervals from bootstrap.
    This function interprets integer and float inputs as the desired
    width of a two-sided confidence interval, and lists and tuples as
    exact quantiles to retrieve.

    Parameters
     ----------
     ci : float, list, tuple
     Determines the quantiles reported for confidence intervals. All entries must
     be in the range [0, 1]. A single float is interpreted as the width of the
     desired confidence interval (e.g. 0.95 is a 95% interval), while a List or Tuple
     reports exact quantiles from the empirical bootstrap distribution
     (e.g. (0, 0.33, 0.5, 0.66, 1) will calculate the min, 33rd, 50th, 66th, and max quantiles).

     method: str
     One of {'percentile', 'bias-corrected'}. Used to validate CI inputs.

     precision: int
     Number of decimals of precision to round quantile calculations to.
    """
    if ci_method not in ["percentile", "bias-corrected"]:
        raise ValueError(
            f"Parameter ci_method: {ci_method} must be set to 'percentile' or"
            " 'bias-corrected'."
        )

    if isinstance(ci, (int, float)) and (0 <= ci <= 1):
        return tuple(np.round((0.5 - ci / 2, 0.5 + ci / 2), decimals=precision))

    elif isinstance(ci, (tuple, list)):
        if all(((0 <= x <= 1) for x in ci)):
            return tuple(ci)
        else:
            raise ValueError(
                f"All desired confidence interval bounds must be within [0, 1]."
            )

    else:
        raise ValueError(
            f"Input to 'ci' of type {type(ci)} must be of type int, float, tuple, or"
            " list."
        )


def create_ci_output(bootstrap_runs, ci, sample_estimate, interval_type="percentile"):
    """Calculates and formats confidence intervals from bootstrap samples."""

    # --- Check if there is significant bias in sample statistics ---
    bootstrap_mean = np.mean(bootstrap_runs, axis=0)

    # Need to use this incredibly odd workaround to support frames with "object" dtypes
    # object typed frames are produced by default from .__group()
    bootstrap_std = np.sqrt(np.var(bootstrap_runs, axis=0, dtype="float64"))
    bias = bootstrap_mean - sample_estimate
    bias_adjusted_mean_estimate = (
        2 * sample_estimate - bootstrap_mean
    )  # Might be useful to publish?

    # Efron and Tibshirani (1993) guidance on identifying "large" bias via bootstrap
    high_bias_regions = np.abs(bias) >= (0.25 * bootstrap_std)
    if np.any(high_bias_regions):
        # TODO: Add bias indication regions into warning. Simply printing all "high_bias_regions" is overwhelming.
        warn(
            f"""Some statistics calculated by MetricFrame are detected to have high bias 
            Consider setting `ci_method` == 'bias-corrected' to mitigate inaccuracies in
            confidence interval calculations."""
        )

    # --- Bootstrap quantile summary calculations ---

    # Adjust which quantiles we sample from the empirical bootstrap distribution if bias correcting
    if interval_type == "bias-corrected":
        # Follows Efron and Tibshirani (1993)
        # with an additional adjustment for overestimating bias in the presence of ties
        num_less = np.sum([x < sample_estimate for x in bootstrap_runs], axis=0)
        num_equal = np.sum([x == sample_estimate for x in bootstrap_runs], axis=0)
        prop_less = (num_less + num_equal / 2) / len(bootstrap_runs)
        z0 = norm.ppf(prop_less)
        z_alphas = norm.ppf(ci)
        adjusted_ci = [norm.cdf(2 * z0 + alpha) for alpha in z_alphas]
    else:
        # Match bias adjusted ci datastructures for downstream code
        adjusted_ci = np.repeat(ci, np.prod(sample_estimate.shape)).reshape(
            (len(ci), *sample_estimate.shape)
        )

    # Potentially slow loop -- cannot seem to vectorize because we need a different set of quantiles for each cell
    # TODO: Look into optimizing this further
    quantiles = []
    for user_ci, q in zip(
        ci, adjusted_ci
    ):  # Calculated based on adjusted_ci values but store with user requested
        bootstrap_quantile = np.zeros_like(sample_estimate)
        for index, index_quantile in np.ndenumerate(q):
            bootstrap_quantile[index] = np.quantile(
                [
                    x.iloc[index] if isinstance(x, (pd.Series, pd.DataFrame)) else x
                    for x in bootstrap_runs
                ],
                q=index_quantile,
                axis=0,
            )

        # Reshape CI outputs to look like original MetricFrame outputs
        if isinstance(sample_estimate, numbers.Number):
            quantiles.append((user_ci, bootstrap_quantile[()]))

        elif isinstance(sample_estimate, pd.Series):
            quantiles.append(
                (user_ci, pd.Series(bootstrap_quantile, index=sample_estimate.index))
            )

        elif isinstance(sample_estimate, pd.DataFrame):
            quantiles.append(
                (
                    user_ci,
                    pd.DataFrame(
                        bootstrap_quantile,
                        index=sample_estimate.index,
                        columns=sample_estimate.columns,
                    ),
                )
            )
        else:
            raise ValueError(
                f"Datatype: {type(sample_estimate)} is unsupported in create_ci_output"
                " bootstrap function. Must be one of type 'Number', 'Pandas Series',"
                " 'Pandas DataFrame'."
            )

    return quantiles
