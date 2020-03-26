# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def assert_n_grid_search_results(n, grid_search_obj):
    assert grid_search_obj._best_grid_index is not None
    assert len(grid_search_obj._predictors) == n
    assert len(grid_search_obj._objectives) == n
    assert len(grid_search_obj._oracle_calls_execution_time) == n
    assert len(grid_search_obj._lambda_vecs.columns) == n
    assert len(grid_search_obj._gammas.columns) == n
