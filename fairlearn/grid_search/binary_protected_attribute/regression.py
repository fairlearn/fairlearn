# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np

from . import utilities


def _regression_weight_function(a_val, trade_off, p0, p1, a0_val):
    if a_val == a0_val:
        return trade_off / p0
    else:
        return (1 - trade_off) / p1


def sweep_bounded_group_loss(learner,
                             x, y, protected_attribute,
                             tradeoffs=None,
                             number_of_tradeoffs=11):
    # Must specify either an array of tradeoffs or the number
    # of tradeoffs to generated
    if bool(tradeoffs is None) == bool(number_of_tradeoffs is None):
        raise RuntimeError("Must specify either tradeoffs "
                           "or number_of_tradeoffs")

    # Extract required statistics from protected_attribute
    p0, p1, a0_val = utilities.generate_protected_attribute_info(
        protected_attribute)

    if tradeoffs is None:
        tradeoffs = np.linspace(0, 1, number_of_tradeoffs)

    result = []
    for trade_off in tradeoffs:
        weight_func = np.vectorize(_regression_weight_function)
        weights = weight_func(protected_attribute,
                              trade_off,
                              p0, p1, a0_val)

        current_learner = copy.deepcopy(learner)
        current_learner.fit(x, y, sample_weight=weights)

        result.append({"model": current_learner,
                       "trade_off": trade_off})
    return result
