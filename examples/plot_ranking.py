# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
=========================================
Ranking
=========================================
"""

from fairlearn.metrics import exposure, utility, exposure_utility_ratio
from fairlearn.metrics import MetricFrame

ranking_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # ranking
sex = ['Male', 'Male', 'Male', 'Female', 'Male', 'Female', 'Female', 'Female', 'Male', 'Female']
y_true = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]  # Continuous relevance score

# Also works with binary relevance scores:
# y_true = [1,1,1,0,1,1,1,0,0,1]  # Binary: Was the recommendation relevant?

# Analyze metrics using MetricFrame
# Careful that in contrast to the classification problem, y_pred now requires a ranking
metrics = {
    'exposure (allocation harm)': exposure,
    'average utility': utility,
    'exposure/utility (quality_of_service)': exposure_utility_ratio
}

mf = MetricFrame(metrics=metrics,
                 y_true=y_true,
                 y_pred=ranking_pred,
                 sensitive_features={'sex': sex})

# Customize the plot
mf.by_group.plot(
    kind="bar",
    subplots=True,
    layout=[1, 3],
    legend=False,
    figsize=(12, 4)
)

# Show the ratio of the metrics, 0 equals unfair and 1 equals fair.
mf.ratio()
