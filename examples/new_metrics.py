# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
==============================
Metrics with Multiple Features
==============================
"""
# %%
# Metrics with Multiple Features
# ==============================
#
# This notebook demonstrates the new API for metrics, which supports
# multiple sensitive and conditional features.

# %%
# Getting the Data
# ----------------
#
# To demonstrate the API, we use the well-known 'Adult' dataset,
# and we train a simple model on it. We start with some
# uncontroversial `import` statements:

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# %%
# Next, we import the data, dropping some of the values to
# help maintain clarity: