# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging

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
