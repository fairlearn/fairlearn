# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from fairlearn.reductions._grid_search import _GridGenerator
from fairlearn.reductions import DemographicParity

from test.unit.reductions.grid_search import _utilities


@pytest.mark.parametrize("grid_size", [10, 11])
def test_grid_generator(grid_size):
    X, y, A = _utilities._quick_data()

    disparity_moment = DemographicParity()
    disparity_moment.load_data(X, y, sensitive_features=A)

    grid = _GridGenerator(grid_size, 2.0, disparity_moment.pos_basis, disparity_moment.neg_basis,
                          disparity_moment.neg_basis_present, False).grid

    expected_index = pd.MultiIndex.from_product([['+', '-'], ['all'], [0, 1]],
                                                names=['sign', 'event', 'group_id'])
    assert (expected_index == grid.index).all()
    expected_grid = pd.DataFrame()
    for i in range(grid_size):
        expected_grid[i] = pd.Series(0.0, index=expected_index)
        expected_grid[i]['-', 'all', 1] = max(2 - 0.4 * i, 0)
        expected_grid[i]['+', 'all', 1] = max(-2 + 0.4 * i, 0)
    assert np.isclose(expected_grid.values, grid.values).all()
