# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""This module contains the functionality for computing metrics
"""

from ._group_metric_result import GroupMetricResult  # noqa: F401
from ._metrics_engine import metric_by_group  # noqa: F401

__all__ = [
    "GroupMetricResult",
    "metric_by_group"
]
