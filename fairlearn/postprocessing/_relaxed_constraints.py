from __future__ import annotations

from collections import deque
from typing import Iterable, Literal

import numpy as np
import pandas as pd


def maximize_objective_with_tolerance(
    dataframes: Iterable[pd.DataFrame],
    weights: Iterable[float],
    tol: float,
    method: Literal["to_overall", "between_groups"] = "between_groups",
    x_col: str = "x",
    y_col: str = "y",
) -> tuple[list[int], float]:
    """
    Finds the indices that maximize the sum of weighted `y` values across multiple DataFrames
    under the constraint that the `x` values satisfy the constraint up to a tolerance `tol`.
    If method is "between_groups", the maximum allowed range of selected `x` values is `tol`.
    If method is "to_overall", the difference between the overall weighted `x` value and the
    smallest and largest selected `x` values must be at most `tol`.

    Args:
        dataframes (Iterable[pd.DataFrame]): An Iterable of DataFrames. Each DataFrame must have:
                                         - An `x` column (sorted in ascending order).
                                         - A `y` column.
        weights (Iterable[float]): An Iterable of weights for each DataFrame.
        tol (float): The maximum allowed tolerance.
        method (Literal["to_overall", "between_groups"]): Whethert the tolerance is computed
            with respect to the overall `x` values or between the groups.

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
    max_objective = float("-inf")
    min_violation = float("inf")

    # Sliding window start pointer
    start = 0

    # Traverse with the end pointer
    for end in range(n):
        # Add the current `end` index to the deques, maintaining descending order of y values
        for k in range(m):
            while deques[k] and y_columns[k][deques[k][-1]] <= y_columns[k][end]:
                deques[k].pop()
            deques[k].append(end)

        # Adjust the window to ensure the tolerance constraint is satisfied
        start = _adjust_window(deques, x_values, weights, start, end, tol, method)

        # Calculate the sum of the maximum weighted `y` values in the current window
        current_indices = [deques[k][0] for k in range(m)]
        current_objective = sum(y_columns[k][current_indices[k]] for k in range(m))

        # Calculate the violation of the equality constraint
        current_violation = _compute_current_violation(current_indices, x_values, weights, method)

        # Update the best solution if the current objective is larger, or if the current objective
        # is equal and the current violation is smaller
        if current_objective > max_objective or (
            current_objective == max_objective and current_violation <= min_violation
        ):
            max_objective = current_objective
            best_indices = current_indices
            min_violation = current_violation

    return best_indices, max_objective


def _adjust_window(
    deques: list[deque[int]],
    x_values: np.ndarray,
    weights: Iterable[float],
    start: int,
    end: int,
    tol: float,
    method: Literal["to_overall", "between_groups"],
) -> int:
    """
    Adjusts the window to ensure the tolerance constraint is satisfied.

    Args:
        deques (list[deque[int]]): A list of deques, each storing indices in descending order of y values.
        x_values (np.ndarray): The array of `x` values.
        weights (Iterable[float]): An Iterable of weights for each DataFrame.
        start (int): The current start index of the window.
        end (int): The current end index of the window.
        tol (float): The maximum allowed range of `x` values for the selected indices.
        method (Literal["to_overall", "between_groups"]): Whethert the tolerance is computed
            with respect to the overall `x` values or between the groups.

    Returns:
        int: The new start index of the window.

    Side effects:
        Modifies the deques in place to remove indices that are no longer in the window.
    """

    if method == "between_groups":
        while x_values[end] - x_values[start] > tol:
            for k in range(len(deques)):
                if deques[k] and deques[k][0] == start:
                    deques[k].popleft()

            start += 1
        return start

    assert method == "to_overall"

    current_indices = [deques[k][0] for k in range(len(deques))]
    current_overall = np.sum(
        [x_values[idx] * list(weights)[k] for k, idx in enumerate(current_indices)]
    )

    while current_overall - x_values[start] > tol or x_values[end] - current_overall > tol:
        for k in range(len(deques)):
            if deques[k] and deques[k][0] == start:
                deques[k].popleft()

        start += 1

        current_indices = [deques[k][0] for k in range(len(deques))]
        current_overall = np.sum(
            [x_values[idx] * list(weights)[k] for k, idx in enumerate(current_indices)]
        )

    return start


def _compute_current_violation(
    current_indices: list[int],
    x_values: np.ndarray,
    weights: Iterable[float],
    method: Literal["to_overall", "between_groups"],
) -> float:
    """
    Computes the violation of the equality constraint for the current window.

    Args:
        current_indices (list[int]): The current indices selected in the window.
        x_values (np.ndarray): The array of `x` values.
        weights (Iterable[float]): An Iterable of weights for each DataFrame.
        method (Literal["to_overall", "between_groups"]): Whethert the tolerance is computed
            with respect to the overall `x` values or between the groups.

    Returns:
        float: The violation of the tolerance constraint.
    """

    min_x = min(x_values[idx] for idx in current_indices)
    max_x = max(x_values[idx] for idx in current_indices)

    if method == "between_groups":
        return max_x - min_x

    assert method == "to_overall"

    current_overall = np.sum(
        [x_values[idx] * list(weights)[k] for k, idx in enumerate(current_indices)]
    )

    return max(current_overall - min_x, max_x - current_overall)
