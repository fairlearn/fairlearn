# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
=================================
Selection rates in census dataset
=================================
"""
# %%
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.datasets import fetch_adult

data = fetch_adult(as_frame=True)
X = data.data
y_true = (data.target == '>50K') * 1
sex = X['sex']

selection_rates = MetricFrame(selection_rate,
                              y_true, y_true,
                              sensitive_features=sex)

selection_rates.by_group.plot(legend=False, title='Fraction earning over $50,000')
