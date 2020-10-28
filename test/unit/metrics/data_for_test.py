# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import numpy as np


y_t = np.asarray(
    [1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1,
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
        0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1]
)

y_p = np.asarray(
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
        1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1]
)

s_w = np.asarray(
    [1, 2, 3, 1, 2, 1, 2, 3, 1, 1, 1, 2, 3, 3, 1, 2, 1, 1, 1, 2, 3,
        2, 2, 1, 2, 1, 3, 1, 3, 1, 4, 1, 2, 1, 3, 2, 3, 3, 4, 1, 4, 1,
        3, 1, 1, 1, 2, 3, 3, 1, 2, 1, 1, 1, 2, 3, 1, 1, 2, 3, 2, 1, 2,
        2, 3, 2, 3, 2, 3, 4, 1]
)


def group_gen(x, runs, groups):
    # For producing arrays such as ['a', 'a', 'b', 'b', 'a', 'a', ... ]
    return groups[(x//runs) % len(groups)]


def array_gen(groups, run_length, num_samples):
    vals = [group_gen(x, run_length, groups) for x in range(num_samples)]
    return np.asarray(vals)


g_1 = array_gen(['aa', 'ba'], 2, len(y_t))
g_2 = array_gen(['f', 'g'], 3, len(y_t))
g_3 = array_gen(['kk', 'm'], 4, len(y_t))
g_4 = array_gen(['pp', 'q'], 5, len(y_t))
