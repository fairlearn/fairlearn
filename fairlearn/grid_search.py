# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This module implements a 1d grid search for binary classification with a single
binary protected attribute.

This is a simplification of the full exponentiated gradient search
and produces a sequence of models, each of which corresponds to a
different accuracy/fairness trade-off
"""

import copy
import numpy as np
import pandas as pd
from collections import namedtuple

def _generate_p0_p1(Y):
    """ Function to compute p0 and p1 for the given
    set of labels Y

    Assume we have already checked that Y is binary with
    values 0 and 1
    """
    unique, counts = np.unique(Y, return_counts=True)

    if len(unique) != 2:
        raise RuntimeError("Supplied Y not binary")

    if not np.array_equal(unique, [0, 1]):
        raise RuntimeError("Supplied Y labels not 0 or 1")

    p0 = counts[0] / len(Y)
    p1 = counts[1] / len(Y)

    return p0, p1

def _generate_weights(Y, A, L, p_ratio):
    W = []

    for y, a in zip(Y, A):
        w = 1e128
        if a==0:
            w = 2*y - 1 - L * p_ratio
        else:
            w = 2*y - 1 + L

        W.append(w)

    return np.array(W)

def classification_binary_protected_1d(learner, X, Y, A, Ls=None, num_Ls = 11):
    """Function to generate a 1d set of models for a classification problem with
    a single binary protected attribute
    """
    # Must specify Ls or num_Ls
    if not( (Ls is None) ^ (num_Ls is None)):
        raise RuntimeError("Must specify either Ls or num_Ls")

    # Check that X, Y and A have the same number of rows
    # Check that Y and A are both 1d vectors
    if len(Y.shape) != 1:
        raise RuntimeError("Supplied Y not 1d vector")

    if len(A.shape) != 1:
        raise RuntimeError("Supplied A not 1d vector")

    if A.shape[0] != Y.shape[0]:
        raise RuntimeError("Supplied A and Y not same length")

    if X.shape[0] != Y.shape[0]:
        raise RuntimeError("Supplied X and Y do not have same number of rows")

    # Check that A only has values 0 and 1
    uniqueA = np.unique(A)
    if not np.array_equal(uniqueA, [0,1]):
        raise RuntimeError("Supplied A labels not 0 or 1")

    # Compute p0 and p1
    # This will also check that Y is binary with values
    # 0 and 1
    p0, p1 = _generate_p0_p1(Y)

    # If not supplied, generate array of trial Lagrangian multipliers
    if Ls is None:
        limit = 1
        if p0/p1 > 1:
            limit = p0/p1
        Ls = np.linspace(-2*limit, 2*limit, num_Ls)

    result = []
    for L in Ls:
        # Generate W array
        W = _generate_weights(Y, A, L, p1/p0)

        # Generate Y'
        f = lambda x: 1 if x>0 else 0
        Yprime = np.vectorize(f)(Y)

        # Run the learner
        mylearner = copy.deepcopy(learner)
        mylearner.fit(X, Yprime, np.absolute(W))

        # Append the new learner, along with its L value to the result
        result.append({"model": mylearner, "lambda":L})

    # Return the result array (tuples of (L,model))
    return result

