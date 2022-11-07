# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import pytest

from fairlearn.reductions import ErrorRate
from fairlearn.reductions._moments.error_rate import _MESSAGE_BAD_COSTS

BAD_COSTS_EXAMPLES = [
    {"fp": 0.0, "fn": 0.0},
    {"fp": -1.0, "fn": 1.0},
    {"fp": 1.0, "fn": -1.0},
    {},
    {"FP": 1.0, "FN": 1.0},
    {"fp": 1.0},
    {"fn": 1.0},
]


@pytest.mark.parametrize("bad_costs", BAD_COSTS_EXAMPLES)
def test_bad_costs(bad_costs):
    with pytest.raises(ValueError) as execInfo:
        _ = ErrorRate(costs=bad_costs)
        assert _MESSAGE_BAD_COSTS in execInfo.value.args[0]
