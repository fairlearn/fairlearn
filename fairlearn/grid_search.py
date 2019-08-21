# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module implements grid searches for fairness/accuracy tradeoffs. Each
point in the grid corresponds to a different accuracy/fairness trade-off. The
dimensionality of the search increases with the number of possible values
for the protected attribute, so for now only binary protected attributes are
considered
"""

import copy
import numpy as np


class BinaryProtectedAttribute:
    def __init__(self):
        # Nothing to do
        return

    def _generate_protected_attribute_info(self, protected_attribute):
        """ Function to compute p0 and p1 for the given
        set of protected_attribute values
        """
        unique_labels, counts = np.unique(
            protected_attribute, return_counts=True)
        if len(unique_labels) > 2:
            raise RuntimeError("Protected Attribute is not binary")

        p0 = counts[0] / len(protected_attribute)
        # It is possible that our data only contain one value for
        # the protected attribute
        # We do not consider this an error
        if len(counts) > 1:
            p1 = counts[1] / len(protected_attribute)
        else:
            p1 = 0

        return p0, p1, unique_labels[0]

    def _weight_function(self, y_val, a_val, L, p_ratio, a0_val):
        if a_val == a0_val:
            return 2 * y_val - 1 - L * p_ratio
        else:
            return 2 * y_val - 1 + L

    def _generate_weights(self, y, protected_attribute, L, p_ratio, a0_val):
        weight_func = np.vectorize(self._weight_function)
        return weight_func(y, protected_attribute, L, p_ratio, a0_val)

    def demographic_parity_binary_classification(self,
                                                 learner,
                                                 x, y, protected_attribute,
                                                 lagrange_multipliers=None,
                                                 number_lagrange_multipliers=11):  # noqa: E501
        """Function to generate a list of models for a binary classification
        problem with a single binary protected attribute. The models sweep
        through different potential Lagrange multipliers for the constrained
        optimisation problem (the constraint being demographic parity), each
        one corresponding to a particular tradeoff between fairness and
        accuracy

        :param learner: An object which can be used to fit a model to
        features, labels and weights. A deep copy of this is made for each
        value of the lagrange multiplier used
        :type learner: Must implement a fit(x, y, sample_weight) method

        :param x: The array of training data features (which may or may
        not contain the protected attribute). Must have as many rows
        as y and protected_attribute
        :type x: Nested list or numpy array with two dimensions or
        pandas dataframe

        :param y: The list of binary classes, which must be 0 or 1. Must
        contain the same number of entries as rows in x. We do not
        consider it an error if only one class is present
        :type y: List or Numpy array with one dimension

        :param protected_attribute: The binary protected attribute
        corresponding to each row of x. We do not consider it an error
        if only one attribute value is present
        :type protected_attribute: List of Numpy array with one dimension

        :param lagrange_multipliers: User specified set of Lagrange
        multipliers to use for the optimisation problem. If this is set
        then number_lagrange_multipliers must be None.
        The result array will be equal in length to this array
        :type lagrange_multipliers: List of real numbers

        :param number_lagrange_multipliers: Specifies the number of
        Lagrange multipliers to use in the optimisation problem. If this
        is set then lagrange_multipliers must be None.
        The result array will have as many entries as specified here

        :return: The models corresponding to each value of the Lagrange
        multiplier tested
        :rtype: List of dictionaries. Each dictionary has fields
        "lagrange_multiplier" and "model." Each model will correspond to
        the input parameter learner after calling 'fit' on it (a deep
        copy is made). The user is responsible for converting these
        objects to an actual model, if further processing is required.
        """
        # Must specify either an array of Lagrange multipliers or how many
        # of them to generate
        if not (lagrange_multipliers is None) ^ (number_lagrange_multipliers is None):  # noqa: E501
            raise RuntimeError("Must specify either lagrange_multipliers "
                               "or number_lagrange_multipliers")

        # Verify we have a binary classification problem
        unique_labels = np.unique(y)
        if len(unique_labels) > 2:
            raise RuntimeError("Supplied Y labels are not binary")

        if not set(unique_labels).issubset({0, 1}):
            raise RuntimeError("Supplied Y labels are not 0 or 1")

        # Extract required statistics from protected_attribute
        p0, p1, a0_val = self._generate_protected_attribute_info(
            protected_attribute)

        # If not supplied, generate array of trial lagrange multipliers
        if lagrange_multipliers is None:
            limit = 1
            if p0 / p1 > 1:
                limit = p0 / p1
            lagrange_multipliers = np.linspace(-2 * limit,
                                               2 * limit,
                                               number_lagrange_multipliers)

        result = []
        for current_multiplier in lagrange_multipliers:
            # Generate weights array
            sample_weights = self._generate_weights(
                y, protected_attribute, current_multiplier, p1 / p0, a0_val)

            # Generate Y'
            def f(x): return 1 if x > 0 else 0
            re_labels = np.vectorize(f)(sample_weights)

            # Run the learner
            current_learner = copy.deepcopy(learner)
            current_learner.fit(
                x, re_labels, sample_weight=np.absolute(sample_weights))

            # Append the new model, along with its current_multiplier value
            # to the result
            # Note that we call it a model because it is a learner which has
            # had 'fit' called
            result.append({"model": current_learner,
                           "lagrange_multiplier": current_multiplier})

        # Return the result array (tuples of (current_multiplier,model))
        return result

    def _regression_weight_function(self, a_val, trade_off, p0, p1, a0_val):
        if a_val == a0_val:
            return trade_off / p0
        else:
            return (1 - trade_off) / p1

    def bounded_group_loss_regression(self,
                                      learner,
                                      x, y, protected_attribute,
                                      tradeoffs=None,
                                      number_of_tradeoffs=11):

        # Extract required statistics from protected_attribute
        p0, p1, a0_val = self._generate_protected_attribute_info(
            protected_attribute)

        if tradeoffs is None:
            tradeoffs = np.linspace(0, 1, number_of_tradeoffs)

        result = []
        for trade_off in tradeoffs:
            weight_func = np.vectorize(self._regression_weight_function)
            weights = weight_func(protected_attribute,
                                  trade_off,
                                  p0, p1, a0_val)

            current_learner = copy.deepcopy(learner)
            current_learner.fit(x, y, sample_weight=weights)

            result.append({"model": current_learner,
                           "trade_off": trade_off})
        return result
