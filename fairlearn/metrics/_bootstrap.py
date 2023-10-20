# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._annotated_metric_function import AnnotatedMetricFunction
from ._disaggregated_result import DisaggregatedResult

logger = logging.getLogger(__name__)

BOOTSTRAP_QUANTILE_ERROR = (
    "Error calling numpy.quantiles. Most likely due to a metric returning a non-scalar"
    " result"
)


def generate_single_bootstrap_sample(
    *,
    random_state: Union[int, np.random.RandomState],
    data: pd.DataFrame,
    annotated_functions: Dict[str, AnnotatedMetricFunction],
    sensitive_feature_names: List[str],
    control_feature_names: Optional[List[str]],
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
    random_state: Optional[Union[int, np.random.RandomState]],
    data: pd.DataFrame,
    annotated_functions: Dict[str, AnnotatedMetricFunction],
    sensitive_feature_names: List[str],
    control_feature_names: Optional[List[str]],
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
        rs = generator.integers(low=0, high=np.iinfo(np.uint32).max, size=n_samples)
    elif isinstance(random_state, np.random.RandomState):
        rs = random_state.randint(low=0, high=np.iinfo(np.uint32).max, size=n_samples)
    elif isinstance(random_state, int):
        generator = np.random.default_rng(seed=random_state)
        rs = generator.integers(low=0, high=np.iinfo(np.uint32).max, size=n_samples)
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


def _calc_series_quantiles(
    *, quantiles: List[float], samples: List[pd.Series]
) -> List[pd.Series]:
    for s in samples:
        assert isinstance(s, pd.Series)
        assert s.name == samples[0].name
        assert all(
            s.index == samples[0].index
        ), "Sanity check shape of bootstrap sample"

    try:
        result_np = np.quantile(samples, q=quantiles, axis=0)
    except ValueError as ve:
        raise ValueError(BOOTSTRAP_QUANTILE_ERROR) from ve
    result = []
    assert result_np.shape[0] == len(quantiles)
    for i in range(result_np.shape[0]):
        nxt = pd.Series(
            name=samples[0].name, index=samples[0].index, data=result_np[i, :]
        )
        result.append(nxt)
    return result


def _calc_dataframe_quantiles(
    *, quantiles: List[float], samples: List[pd.DataFrame]
) -> List[pd.DataFrame]:
    for s in samples:
        assert isinstance(s, pd.DataFrame)
        assert all(s.columns == samples[0].columns)
        assert all(
            s.index == samples[0].index
        ), "Sanity check shape of bootstrap sample"

    try:
        result_np = np.quantile(samples, q=quantiles, axis=0)
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


def calculate_pandas_quantiles(
    quantiles: List[float], bootstrap_samples: List[Union[pd.Series, pd.DataFrame]]
) -> Union[List[pd.Series], List[pd.DataFrame]]:
    """Calculate quantiles for a list of pandas objects."""
    if isinstance(bootstrap_samples[0], pd.Series):
        result = _calc_series_quantiles(quantiles=quantiles, samples=bootstrap_samples)
    elif isinstance(bootstrap_samples[0], pd.DataFrame):
        result = _calc_dataframe_quantiles(
            quantiles=quantiles, samples=bootstrap_samples
        )
    else:
        assert False, "Should not be possible to get here"
    return result
