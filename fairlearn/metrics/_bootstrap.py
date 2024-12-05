# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.
from __future__ import annotations

import logging
from functools import reduce
from typing import List

import numpy as np
import pandas as pd

from ._annotated_metric_function import AnnotatedMetricFunction
from ._disaggregated_result import DisaggregatedResult

logger = logging.getLogger(__name__)

BOOTSTRAP_QUANTILE_ERROR = (
    "Error calling numpy.quantiles. Most likely due to a metric returning a non-scalar result"
)


def generate_single_bootstrap_sample(
    *,
    random_state: int | np.random.RandomState,
    data: pd.DataFrame,
    annotated_functions: dict[str, AnnotatedMetricFunction],
    sensitive_feature_names: list[str],
    control_feature_names: list[str] | None = None,
) -> DisaggregatedResult:
    """Create a single bootstrapped DisaggregatedResult."""
    assert random_state is not None, "Must specify random_state"

    sampled_data = data.sample(
        frac=1, replace=True, random_state=random_state, axis=0, ignore_index=True
    )

    result = DisaggregatedResult.create(
        data=sampled_data,
        annotated_functions=annotated_functions,
        sensitive_feature_names=sensitive_feature_names,
        control_feature_names=control_feature_names,
    )
    return result


def generate_bootstrap_samples(
    *,
    n_samples: int,
    random_state: int | np.random.RandomState | None,
    data: pd.DataFrame,
    annotated_functions: dict[str, AnnotatedMetricFunction],
    sensitive_feature_names: list[str],
    control_feature_names: list[str] | None = None,
) -> List[DisaggregatedResult]:
    """Create a list of bootstrapped DisaggregatedResults.

    The list will contain n_samples, and a random_state must be supplied.

    If random_state is an integer or a numpy generator, it will be used to
    create a list of num_samples random integers, which will then be used as
    the seeds for each bootstrap sample
    """
    assert n_samples >= 1
    if random_state is None:
        generator = np.random.default_rng()
        rs = generator.integers(
            low=0, high=np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32
        )
    elif isinstance(random_state, np.random.RandomState):
        rs = random_state.randint(
            low=0, high=np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32
        )
    elif isinstance(random_state, int):
        generator = np.random.default_rng(seed=random_state)
        rs = generator.integers(
            low=0, high=np.iinfo(np.uint32).max, size=n_samples, dtype=np.uint32
        )
    else:
        raise ValueError(f"Unsupported random_state: {random_state}")

    result = []
    for i in range(n_samples):
        nxt = generate_single_bootstrap_sample(
            random_state=rs[i],
            data=data,
            annotated_functions=annotated_functions,
            sensitive_feature_names=sensitive_feature_names,
            control_feature_names=control_feature_names,
        )
        result.append(nxt)

    return result


def _calc_series_quantiles(*, quantiles: list[float], samples: list[pd.Series]) -> list[pd.Series]:
    for s in samples:
        assert isinstance(s, pd.Series)
        assert s.name == samples[0].name
        assert all(s.index == samples[0].index), "Sanity check shape of bootstrap sample"

    try:
        result_np = np.quantile(samples, q=quantiles, axis=0)
    except ValueError as ve:
        raise ValueError(BOOTSTRAP_QUANTILE_ERROR) from ve
    result = []
    assert result_np.shape[0] == len(quantiles)
    for i in range(result_np.shape[0]):
        nxt = pd.Series(name=samples[0].name, index=samples[0].index, data=result_np[i, :])
        result.append(nxt)
    return result


def _calc_dataframe_quantiles(
    *, quantiles: list[float], samples: list[pd.DataFrame]
) -> list[pd.DataFrame]:
    samples = _align_sample_indices(samples)

    for s in samples:
        assert isinstance(s, pd.DataFrame)
        assert all(s.columns == samples[0].columns)
        assert all(s.index == samples[0].index), "Sanity check shape of bootstrap sample"

    try:
        result_np = np.nanquantile(samples, q=quantiles, axis=0)
    except ValueError as ve:
        raise ValueError(BOOTSTRAP_QUANTILE_ERROR) from ve

    result = []
    assert result_np.shape[0] == len(quantiles)
    for i in range(result_np.shape[0]):
        nxt = pd.DataFrame(
            columns=samples[0].columns, index=samples[0].index, data=result_np[i, :, :]
        )
        result.append(nxt)

    return result


def _align_sample_indices(samples: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """
    Align the indices of the bootstrapped DataFrames, so that all the combinations of sensitive
    and control feature values are present in all the samples.

    This is achieved by reindexing them to a common outer union of all indices, filling the missing
    combinations with NaNs.

    If a combination is missing from all the samples, it won't appear in the common index.

    Parameters
    ----------
    samples : list[pd.DataFrame]
        A list of pandas DataFrames to be aligned.

    Returns
    -------
    list[pd.DataFrame]
        A list of pandas DataFrames with aligned indices.
    """
    all_indices = [sample.index for sample in samples]
    outer_common_index = reduce(lambda x, y: x.union(y), all_indices)
    samples = [sample.reindex(outer_common_index) for sample in samples]
    return samples


def calculate_pandas_quantiles(
    quantiles: list[float], bootstrap_samples: list[pd.Series] | list[pd.DataFrame]
) -> list[pd.Series] | list[pd.DataFrame]:
    """Calculate quantiles for a list of pandas objects."""
    if isinstance(bootstrap_samples[0], pd.Series):
        result = _calc_series_quantiles(quantiles=quantiles, samples=bootstrap_samples)
    elif isinstance(bootstrap_samples[0], pd.DataFrame):
        result = _calc_dataframe_quantiles(quantiles=quantiles, samples=bootstrap_samples)
    else:
        assert False, "Should not be possible to get here"
    return result
