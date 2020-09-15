# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from fairlearn.reductions._grid_search.grid_search import _GridGenerator
from fairlearn.reductions import DemographicParity, EqualizedOdds
from fairlearn.reductions._moments.utility_parity import _ALL, _SIGN, _EVENT, \
    _GROUP_ID

from test.unit.reductions.grid_search.utilities import _quick_data


@pytest.mark.parametrize("grid_size", [5, 6, 10, 11, 100, 101, 102])
@pytest.mark.parametrize("grid_limit", [1.0, 2.0, 4.0, 10.0])
def test_grid_generator_demographic_parity(grid_size, grid_limit):
    disparity_moment = DemographicParity()
    events = [_ALL]

    grid = calculate_grid(grid_limit, grid_size, disparity_moment)

    expected_index = pd.MultiIndex.from_product([['+', '-'], events, [0, 1]],
                                                names=[_SIGN, _EVENT, _GROUP_ID])
    assert (expected_index == grid.index).all()
    expected_grid = pd.DataFrame()
    grid_size_or_next_smaller_even_number = int(grid_size / 2) * 2
    step_size = 2 * grid_limit / grid_size_or_next_smaller_even_number
    for i in range(grid_size):
        expected_grid[i] = pd.Series(0.0, index=expected_index)
        expected_grid[i]['-', _ALL, 1] = max(grid_limit - step_size * i, 0)
        expected_grid[i]['+', _ALL, 1] = max(-grid_limit + step_size * i, 0)
    assert np.isclose(expected_grid.values, grid.values).all()


@pytest.mark.parametrize("grid_size", [5, 6])
@pytest.mark.parametrize("grid_limit", [1.0, 2.0])
@pytest.mark.parametrize("grid_offset", [[0, 0.2, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0],
                                         [0, 0, 0, 0.2]])
def test_grid_generator_demographic_parity_with_center(grid_size, grid_limit, grid_offset):
    disparity_moment = DemographicParity()
    events = [_ALL]

    _index = pd.MultiIndex.from_product([['+', '-'], events, [0, 1]],
                                        names=[_SIGN, _EVENT, _GROUP_ID])
    grid_offset = pd.Series(grid_offset, _index)
    grid = calculate_grid(grid_limit, grid_size, disparity_moment, grid_offset)

    expected_index = pd.MultiIndex.from_product([['+', '-'], events, [0, 1]],
                                                names=[_SIGN, _EVENT, _GROUP_ID])
    assert (expected_index == grid.index).all()
    expected_grid = pd.DataFrame()
    grid_size_or_next_smaller_even_number = int(grid_size / 2) * 2
    step_size = 2 * grid_limit / grid_size_or_next_smaller_even_number
    for i in range(grid_size):
        expected_grid[i] = pd.Series(0.0, index=expected_index)
        expected_grid[i]['-', _ALL, 1] = max(grid_limit - step_size * i, 0) + \
            grid_offset['-', _ALL, 1]
        expected_grid[i]['+', _ALL, 1] = max(-grid_limit + step_size * i, 0) + \
            grid_offset['+', _ALL, 1]
    assert np.isclose(expected_grid.values, grid.values).all()


@pytest.mark.parametrize("grid_limit", [0.1, 0.5, 1, 2, 3, 10, 99, 100, 101])
def test_grid_generator_equalized_odds_basic(grid_limit):
    # Equalized odds has four rows with potential non-zero values in the grid.
    # grid_size = 5 ensures that each of the groups have their own column.
    grid_size = 5

    disparity_moment = EqualizedOdds()
    label0 = 'label=0'
    label1 = 'label=1'
    events = [label0, label1]

    grid = calculate_grid(grid_limit, grid_size, disparity_moment)

    expected_index = pd.MultiIndex.from_product([['+', '-'], events, [0, 1]],
                                                names=[_SIGN, _EVENT, _GROUP_ID])
    assert (expected_index == grid.index).all()
    expected_grid = pd.DataFrame()
    for i in range(grid_size):
        expected_grid[i] = pd.Series(0.0, index=expected_index)

    expected_grid[0]['-', label0, 1] = grid_limit
    expected_grid[1]['-', label1, 1] = grid_limit
    expected_grid[3]['+', label1, 1] = grid_limit
    expected_grid[4]['+', label0, 1] = grid_limit

    assert np.isclose(expected_grid.values, grid.values).all()


@pytest.mark.parametrize("grid_limit", [0.1, 0.5, 1, 2, 5, 10, 99, 100])
@pytest.mark.parametrize("grid_size", [13])
def test_grid_generator_equalized_odds(grid_limit, grid_size):
    # Equalized odds has four rows with potential non-zero values in the grid.
    # With grid_size = 13 we get exactly one column with the grid_limit value per row,
    # one column with half the grid_limit value per row, and combinations of rows

    disparity_moment = EqualizedOdds()
    label0 = 'label=0'
    label1 = 'label=1'
    events = [label0, label1]

    grid = calculate_grid(grid_limit, grid_size, disparity_moment)

    expected_index = pd.MultiIndex.from_product([['+', '-'], events, [0, 1]],
                                                names=[_SIGN, _EVENT, _GROUP_ID])
    assert (expected_index == grid.index).all()
    expected_grid = pd.DataFrame()
    for i in range(grid_size):
        expected_grid[i] = pd.Series(0.0, index=expected_index)

    gl = grid_limit  # abbreviation for readibility
    expected_grid.loc['+', label0, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, gl/2, gl/2, gl/2, gl]
    expected_grid.loc['+', label1, 1] = [0, 0, 0, gl/2, 0, 0, 0, gl/2, gl, 0, 0, gl/2, 0]
    expected_grid.loc['-', label0, 1] = [gl, gl/2, gl/2, gl/2, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_grid.loc['-', label1, 1] = [0, gl/2, 0, 0, gl, gl/2, 0, 0, 0, gl/2, 0, 0, 0]

    assert np.isclose(expected_grid.values, grid.values).all()


@pytest.mark.parametrize("grid_limit", [0.1, 2])
@pytest.mark.parametrize("grid_size", [13])
@pytest.mark.parametrize("grid_offset", [[0, 0, 0, 0, 0, 0, 0, 0.2], [0, 0, 0, 1, 0, 0, 0, 0],
                                         [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0.2, 0, 0]])
def test_grid_generator_equalized_odds_with_center(grid_limit, grid_size, grid_offset):
    # Equalized odds has four rows with potential non-zero values in the grid.
    # With grid_size = 13 we get exactly one column with the grid_limit value per row,
    # one column with half the grid_limit value per row, and combinations of rows

    disparity_moment = EqualizedOdds()
    label0 = 'label=0'
    label1 = 'label=1'
    events = [label0, label1]

    _index = pd.MultiIndex.from_product([['+', '-'], events, [0, 1]],
                                        names=[_SIGN, _EVENT, _GROUP_ID])
    grid_offset = pd.Series(grid_offset, index=_index)
    grid = calculate_grid(grid_limit, grid_size, disparity_moment, grid_offset)

    expected_index = pd.MultiIndex.from_product([['+', '-'], events, [0, 1]],
                                                names=[_SIGN, _EVENT, _GROUP_ID])
    assert (expected_index == grid.index).all()
    expected_grid = pd.DataFrame()
    for i in range(grid_size):
        expected_grid[i] = pd.Series(0.0, index=expected_index)

    gl = grid_limit  # abbreviation for readibility
    expected_grid.loc['+', label0, 1] = [0, 0, 0, 0, 0, 0, 0, 0, 0, gl/2, gl/2, gl/2, gl] + \
        grid_offset['+', label0, 1]
    expected_grid.loc['+', label1, 1] = [0, 0, 0, gl/2, 0, 0, 0, gl/2, gl, 0, 0, gl/2, 0] + \
        grid_offset['+', label1, 1]
    expected_grid.loc['-', label0, 1] = [gl, gl/2, gl/2, gl/2, 0, 0, 0, 0, 0, 0, 0, 0, 0] + \
        grid_offset['-', label0, 1]
    expected_grid.loc['-', label1, 1] = [0, gl/2, 0, 0, gl, gl/2, 0, 0, 0, gl/2, 0, 0, 0] + \
        grid_offset['-', label1, 1]

    assert np.isclose(expected_grid.values, grid.values).all()


def calculate_grid(grid_limit, grid_size, disparity_moment, grid_offset=None):
    X, y, A = _quick_data()

    disparity_moment.load_data(X, y, sensitive_features=A)

    return _GridGenerator(grid_size, grid_limit,
                          disparity_moment.pos_basis, disparity_moment.neg_basis,
                          disparity_moment.neg_basis_present, False, grid_offset).grid
