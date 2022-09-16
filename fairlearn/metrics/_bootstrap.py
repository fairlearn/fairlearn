# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging
from re import A

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._annotated_metric_function import AnnotatedMetricFunction
from ._disaggregated_result import DisaggregatedResult

logger = logging.getLogger(__name__)


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
    random_state: Union[
        int, np.random.RandomState, List[int], List[np.random.RandomState]
    ],
    data: pd.DataFrame,
    annotated_functions: Dict[str, AnnotatedMetricFunction],
    sensitive_feature_names: List[str],
    control_feature_names: Optional[List[str]],
) -> List[DisaggregatedResult]:
    """Create a list of bootstrapped DisaggregatedResults.

    The list will contain n_samples, and a random_state must be supplied.

    If random_state is a list (assumed to be of integers of numpy generators),
    then it must be of length n_samples, and each will be used for the corresponding
    sample.

    If random_state is an integer or a numpy generator, it will be used to
    create a list of num_samples random integers, which will then be used as
    the seeds for each bootstrap sample
    """
    assert n_samples >= 1
    assert random_state is not None, "Must specify random_state"
    if isinstance(random_state, list):
        assert len(random_state) == n_samples, "Must have one state per desired sample"
        rs = random_state
    else:
        if isinstance(random_state, np.random.RandomState):
            rs = random_state.randint(
                low=0, high=np.iinfo(np.uint32).max, size=n_samples
            )
        else:
            generator = np.random.default_rng(seed=random_state)
            rs = generator.integers(low=0, high=np.iinfo(np.uint32).max, size=n_samples)

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


def _calc_series_quantiles(*, quantiles: List[float], samples: List[pd.Series]):
    for s in samples:
        assert isinstance(s, pd.Series)
        assert s.name == samples[0].name
        assert all(s.index == samples[0].index)

    result_np = np.quantile(samples, q=quantiles, axis=0)
    result = pd.Series(
        name=samples[0].name,
        index=samples[0].index,
        data=[result_np[:, i] for i in range(result_np.shape[1])],
    )
    return result


def _calc_dataframe_quantiles(*, quantiles: List[float], samples: List[pd.DataFrame]):
    for s in samples:
        assert isinstance(s, pd.DataFrame)
        assert all(s.columns == samples[0].columns)
        assert all(s.index == samples[0].index)

    result_np = np.quantile(samples, q=quantiles, axis=0)

    result_data = dict()
    for i_c, c in enumerate(samples[0].columns):
        nxt = []
        for r in range(len(samples[0].index)):
            qs = result_np[:, r, i_c]
            nxt.append(qs)
        result_data[c] = nxt
    result = pd.DataFrame(
        columns=samples[0].columns,
        index=samples[0].index,
        data=result_data,
    )

    return result


def calculate_pandas_quantiles(
    quantiles: List[float], bootstrap_samples: List[Union[pd.Series, pd.DataFrame]]
) -> Union[pd.Series, pd.DataFrame]:
    """Calculate quantiles for a list of pandas objects"""

    if isinstance(bootstrap_samples[0], pd.Series):
        result = _calc_series_quantiles(quantiles=quantiles, samples=bootstrap_samples)
    elif isinstance(bootstrap_samples[0], pd.DataFrame):
        result = _calc_dataframe_quantiles(
            quantiles=quantiles, samples=bootstrap_samples
        )
    else:
        assert False
    return result
