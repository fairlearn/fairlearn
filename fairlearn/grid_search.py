# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module implements a 1d grid search for cases with a single
binary protected attribute. The grid search produces a sequence of models,
each of which corresponds to a different accuracy/fairness trade-off. The
dimensionality of the search increases with the number of possible values
for the protected attribute, so this technique does not scale beyond a
binary protected attribute.
"""

import copy
import numpy as np

class BinaryClassificationGridSearch:
    def __init__(self):
        # Nothing to do
        return

    def _generate_p0_p1(self, y):
        """ Function to compute p0 and p1 for the given
        set of labels
        """
        unique, counts = np.unique(y, return_counts=True)

        if len(unique) != 2:
            raise RuntimeError("Supplied y not binary")

        if not np.array_equal(unique, [0, 1]):
            raise RuntimeError("Supplied y labels not 0 or 1")

        p0 = counts[0] / len(y)
        p1 = counts[1] / len(y)

        return p0, p1

    def _weight_function(self, y_val, a_val, L, p_ratio):
        if a_val == 0:
            return 2*y_val - 1 - L * p_ratio
        else:
            return 2*y_val - 1 + L

    def _generate_weights(self, y, protected_attribute, L, p_ratio):
        weight_func = np.vectorize(self._weight_function)
        return weight_func(y, protected_attribute, L, p_ratio)

    def grid_search_binary_protected_attribute(self, learner, x, y, protected_attribute, lagrangian_multipliers=None, number_lagrangian_multipliers=11):
        """Function to generate a list of models for a binary classification problem with
        a single binary protected attribute. The models sweep through different potential
        Lagrangian multipliers for the constrained optimisation problem (the constraint
        being demographic parity), each one corresponding to a particular tradeoff between
        fairness and accuracy

        :param learner: An object which can be used to fit a model to features, labels and
        weights. A deep copy of this is made for each value of the Lagrangian multiplier used
        :type learner: Must implement a fit(x, y, sample_weight) method

        :param x: The array of training data features (which may or may not contain the
        protected attribute). Must have as many rows as y and protected_attribute
        :type x: Numpy array with two dimensions or pandas dataframe

        :param y: The list of binary classes, which must be 0 or 1. Must contain the same
        number of entries as rows in x
        :type y: Numpy array with one dimension

        :param protected_attribute: The protected attribute corresponding to each row of x.
        Must be either 0 or 1
        :type protected_attribute: Numpy array with one dimension
        
        :param lagrangian_multipliers: User specified set of Lagrangian multipliers to use for
        the optimisation problem. If this is set then num_Ls must be None. The result array will
        be equal in length to this array
        :type lagrangian_multipliers: List of real numbers
        
        :param number_lagrangian_multipliers: Specifies the number of Lagrangian multipliers to
        use in the optimisation problem. If this is set then lagrangian_multipliers must be None.
        The result array will have as many entries as specified here

        :return: The models corresponding to each value of the Lagrangian multiplier tested
        :rtype: List of dictionaries. Each dictionary has fields "lagrangian_multiplier" and "model." Each
        model will correspond to the input parameter learner after calling 'fit' on it (a deep
        copy is made). The user is responsible for converting these objects to an actual model,
        if further processing is required.
        """
        # Must specify either Ls or num_Ls
        if not (lagrangian_multipliers is None) ^ (number_lagrangian_multipliers is None):
            raise RuntimeError("Must specify either lagrangian_multipliers or number_lagrangian_multipliers")

        # Check that x, y and protected_attribute have the same number of rows
        # Check that y and protected_attribute are both 1d vectors
        if len(y.shape) != 1:
            raise RuntimeError("Supplied y not 1d vector")

        if len(protected_attribute.shape) != 1:
            raise RuntimeError("Supplied protected_attribute not 1d vector")

        if protected_attribute.shape[0] != y.shape[0]:
            raise RuntimeError("Supplied protected_attribute and y not same length")

        if x.shape[0] != y.shape[0]:
            raise RuntimeError("Supplied x and y do not have same number of rows")

        # Check that A only has values 0 and 1
        unique_protected_attribute_values = np.unique(protected_attribute)
        if not np.array_equal(unique_protected_attribute_values, [0, 1]):
            raise RuntimeError("Supplied protected_attribute labels not 0 or 1")

        # Compute p0 and p1
        # This will also check that Y is binary with values
        # 0 and 1
        p0, p1 = self._generate_p0_p1(y)

        # If not supplied, generate array of trial Lagrangian multipliers
        if lagrangian_multipliers is None:
            limit = 1
            if p0/p1 > 1:
                limit = p0/p1
            lagrangian_multipliers = np.linspace(-2*limit, 2*limit, number_lagrangian_multipliers)

        result = []
        for current_multiplier in lagrangian_multipliers:
            # Generate weights array
            sample_weights = self._generate_weights(y, protected_attribute, current_multiplier, p1/p0)

            # Generate Y'
            f = lambda x: 1 if x > 0 else 0
            re_labels = np.vectorize(f)(sample_weights)

            # Run the learner
            mylearner = copy.deepcopy(learner)
            mylearner.fit(x, re_labels, sample_weight=np.absolute(sample_weights))

            # Append the new learner, along with its current_multiplier value to the result
            result.append({"model": mylearner, "lagrangian_multiplier":current_multiplier})

        # Return the result array (tuples of (current_multiplier,model))
        return result
