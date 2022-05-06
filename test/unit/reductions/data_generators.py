# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import random

import numpy as np
import pandas as pd


def loan_scenario_generator(
    class_sizes,
    success_fractions,
    sensitive_feature_classes,
    control_feature_classes,
    seed=1632753,
):
    random.seed(seed)
    IB = []
    SF = []
    PLOAN = []
    Y = []

    for ib in control_feature_classes:
        for sf in sensitive_feature_classes:
            n_curr = class_sizes[ib][sf]
            f_curr = success_fractions[ib][sf]

            for i in range(n_curr):
                IB.append(ib)
                SF.append(sf)
                flip = random.random()
                PLOAN.append(flip)
                Y.append(1 if flip < f_curr else 0)

    X = pd.DataFrame(
        data=np.transpose([IB, SF, PLOAN]), columns=["ctrl", "sens", "ploan"]
    )

    return X, Y
