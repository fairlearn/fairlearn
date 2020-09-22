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


def _group_gen(x, runs, groups):
    # For producing arrays such as ['a', 'a', 'b', 'b', 'a', 'a', ... ]
    assert runs >= len(groups)
    return groups[(x//runs) % len(groups)]


g_1 = np.asarray([_group_gen(x, 2, ['aa', 'ba']) for x in range(len(y_t))])
g_2 = np.asarray([_group_gen(x, 3, ['f', 'g']) for x in range(len(y_t))])
g_3 = np.asarray([_group_gen(x, 4, ['kk', 'm']) for x in range(len(y_t))])
g_4 = np.asarray([_group_gen(x, 5, ['pp', 'q']) for x in range(len(y_t))])
