# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" This module contains the functionality for computing metrics
"""

from ._group_metric_result import GroupMetricResult  # noqa: F401
from .metrics_engine import metric_by_group, make_group_metric  # noqa: F401
