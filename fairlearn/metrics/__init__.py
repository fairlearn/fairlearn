# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sklearn.metrics as skm

from .metric import DisparityMetric   # noqa: F401
from .demographic_parity import DemographicParity  # noqa: F401
from .equalized_odds import EqualizedOdds  # noqa: F401
from .bounded_group_loss import BoundedGroupLoss  # noqa: F401

from .group_metric_result import GroupMetricResult  # noqa: F401
from .metrics_engine import metric_by_group, make_group_metric  # noqa: F401

group_accuracy_score = make_group_metric(skm.accuracy_score)
group_precision_score = make_group_metric(skm.precision_score)
group_recall_score = make_group_metric(skm.recall_score)
