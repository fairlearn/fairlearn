from __future__ import annotations

from collections import deque
from typing import Iterable

import pandas as pd


def maximize_objective_with_tolerance(
    dataframes: Iterable[pd.DataFrame],
    weights: Iterable[float],
    tol: float,
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[list[int], float]:
    """
    Finds the indices that maximize the sum of weighted `y` values across multiple DataFrames
    under the constraint that the difference between the maximum and minimum `x` values
    in the selected indices is at most `tol`.

    Args:
        dataframes (Iterable[pd.DataFrame]): An Iterable of DataFrames. Each DataFrame must have:
                                         - An `x` column (sorted in ascending order).
                                         - A `y` column.
        weights (Iterable[float]): An Iterable of weights for each DataFrame `y` column.
        tol (float): The maximum allowed range of `x` values for the selected indices.

    Returns:
        list[int], float: - A list of selected indices (one per DataFrame).
                          - The maximum sum of weighted `y` values under the constraint.
    """
    # Extract `x` values (all DataFrames must share the same `x` values)
    x_values = dataframes[0][x_col].values
    n = len(x_values)

    # Extract weighted `y` columns
    y_columns = [df[y_col].values * weight for df, weight in zip(dataframes, weights)]
    m = len(y_columns)

    # Initialize a deque for each DataFrame to store indices in descending order of y values
    deques = [deque() for _ in range(m)]

    # Variables to track the best solution
    best_indices = []
    max_sum = float("-inf")
    min_range = float("inf")

    # Sliding window start pointer
    start = 0

    # Traverse with the end pointer
    for end in range(n):
        # Add the current `end` index to the deques, maintaining descending order of y values
        for k in range(m):
            while deques[k] and y_columns[k][deques[k][-1]] <= y_columns[k][end]:
                deques[k].pop()
            deques[k].append(end)

        # Adjust the window to ensure the range of x values is within tol
        if x_values[end] - x_values[start] > tol:
            for k in range(m):
                if deques[k] and deques[k][0] == start:
                    deques[k].popleft()
            start += 1

        # Calculate the sum of the maximum weighted `y` values in the current window
        current_indices = [deques[k][0] for k in range(m)]
        current_sum = sum(y_columns[k][current_indices[k]] for k in range(m))
        # Calculate the range of x values in the selected indices
        current_range = max(x_values[idx] for idx in current_indices) - min(
            x_values[idx] for idx in current_indices
        )

        # Update the best solution if the current sum is larger, or if the current sum is equal
        # and the current range is smaller
        if current_sum > max_sum or (current_sum == max_sum and current_range <= min_range):
            max_sum = current_sum
            best_indices = current_indices
            min_range = current_range

    return best_indices, max_sum
