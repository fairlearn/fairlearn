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
    """Function to generate a list of models for a binary
    regression problem with a single binary protected attribute.
    The trade off parameter controls how much weight is put on
    the '0' labels of the protected attribute vs. the '1' labels
    of the attribute. The trade off parameter varies between
    zero and one, the extremes corresponding to performing the
    regression only with the specified label. This sequence
    of models enables selection for the best bounded group loss.

    :param learner: An object which can be used to fit a model to
    features, labels and weights. A deep copy of this is made for each
    value of the lagrange multiplier used
    :type learner: Must implement a fit(x, y, sample_weight) method

    :param x: The array of training data features (which may or may
    not contain the protected attribute). Must have as many rows
    as y and protected_attribute
    :type x: Nested list or numpy array with two dimensions or
    pandas dataframe

    :param y: The true regression values. Must contain the same
    number of entries as rows in x.
    :type y: List or Numpy array with one dimension or pandas Series

    :param protected_attribute: The binary protected attribute
    corresponding to each row of x. We do not consider it an error
    if only one attribute value is present
    :type protected_attribute: List or Numpy array with one
    dimension or pandas Series

    :param tradeoffs: User specified set of trade offs to use for
    the sweep. If this is set then number_of_tradeoffs must be None.
    The result array will be equal in length to this array
    :type tradeoffs: List of real numbers from [0, 1]

    :param number_of_tradeoffs: Specifies the number of trade offs
    to use for this sweep. They will be automatically generated in
    the range [0, 1]. If this is set then tradeoffs must be None.
    The result array will have as many entries as specified here

    :return: The models corresponding to each value of the trade offs
    tested
    :rtype: List of dictionaries. Each dictionary has fields
    "tradeoff" and "model." Each model will correspond to
    the input parameter learner after calling 'fit' on it (a deep
    copy is made). The user is responsible for converting these
    objects to an actual model, if further processing is required.
    """
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
                       "tradeoff": trade_off})
    return result
