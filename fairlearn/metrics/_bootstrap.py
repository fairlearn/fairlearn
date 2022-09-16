# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ._annotated_metric_function import AnnotatedMetricFunction
from ._disaggregated_result import DisaggregatedResult

logger = logging.getLogger(__name__)


def generate_bootstrapped_metrics(
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
