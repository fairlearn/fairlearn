# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def _quick_data(A_two_dim=False):
    # Data are random and do not matter for these tests
    feature_1 = [0, 1, 2, 3, 4, 5, 6, 7]
    feature_2 = [5, 4, 3, 2, 7, 8, 3, 4]
    feature_3 = [9, 2, 4, 2, 9, 3, 1, 8]
    X = np.stack((feature_1, feature_2, feature_3), -1)
    Y = np.array([0, 1, 0, 1, 1, 1, 1, 0])
    A = np.array([1, 0, 0, 0, 0, 1, 1, 1])
    if A_two_dim:
        # Grid Search is still restricted to binary sensitive features.
        # Even though we provide multiple columns of sensitive features,
        # the merged feature comprised of these columns can only have
        # two unique values.
        A = np.stack((A, A), -1)
    return X, Y, A


def assert_n_grid_search_results(n, grid_search_obj):
    assert grid_search_obj._best_grid_index is not None
    assert len(grid_search_obj._predictors) == n
    assert len(grid_search_obj._objectives) == n
    assert len(grid_search_obj._oracle_execution_times) == n
    assert len(grid_search_obj._lambda_vecs.columns) == n
    assert len(grid_search_obj._gammas.columns) == n
