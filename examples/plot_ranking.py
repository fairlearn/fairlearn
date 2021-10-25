# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
=========================================
Ranking
=========================================
"""

from fairlearn.metrics import exposure, utility, exposure_utility_ratio
from fairlearn.metrics import MetricFrame

# %%
# This notebook shows how to use Fairlearn with rankings. We showcase the example "Fairly
# Allocating Economic Opportunity" from the paper "Fairness of Exposure in Ranking" by Singh and
# Joachims (2018).
# The example demonstrates how small differences in item relevance can lead to large differences
# in exposure.
#
# Consider a web-service that connects employers (users) to potential employees (items).
# The web-service uses a ranking-kased system to present a set of 6 applicants of which 3 are male
# and 3 are female. Male applicants have relevance of 0.80, 0.79, 0.78 respectively for the
# employer, while female applicants have relevance of 0.77, 0.76, 0.75.
# In this setting a relevance of 0.75 is defined as, 75% of all employers issuing the query found
# the applicant relevant.
#
# The Probability Ranking Principle suggests to rank the applicants in decreasing order of
# relevance. What does this mean for the exposure between the two groups?

ranking_pred = [1, 2, 3, 4, 5, 6]  # ranking
sex = ['Male', 'Male', 'Male', 'Female', 'Female', 'Female']
y_true = [0.82, 0.81, 0.80, 0.79, 0.78, 0.77]

# %%
# Here we define what metrics we want to analyze.
#
# - The `exposure` metric shows the average exposure that each group gets, based on their position
#   biases. Exposure is the value that we assign to every place in the ranking, calculated by a
#   standard exposure drop-off of :math:`1/log_2(1+j)` as used in Discounted Cumulative Gain (DCG),
#   to account for position bias. If there are big differences in exposure
#   we could say that there is allocation harm in the data, i.e. males are on average ranked way
#   higher than females by the web-service.
#
# - The `utility` metric shows the average relevance that each group has.
#
# - The `exposure_utility_ratio` metric shows quality-of-service harms in the data. Since it shows
#   what the average exposure of each group is compared to its relevance. If there a big
#   differences in this metric we could say that the exposure of some sensitive groups is not
#   proportional to its utility.

metrics = {
    'exposure (allocation harm)': exposure,
    'average utility': utility,
    'exposure/utility (quality-of-service)': exposure_utility_ratio
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

# %%
# The first plot shows that the web-service that men get significantly more exposure than women.
# Although the second plot shows that the utility of females is comparable to the males group.
# Therefor we can say that the ranking contains quality-of-service harm against women, since the
# exposure/utility ratio is not equal (plot 3)

# %%
# How can we fix this? A simple solution is to rerank the items, in such a way that females get
# more exposure and males get less exposure. For example we can switch the top male with the top
# female applicant and remeasure the quality-of-service harm.

ranking_pred = [1, 2, 3, 4, 5, 6]  # ranking
sex = ['Female', 'Male', 'Male', 'Male', 'Female', 'Female']
y_true = [0.79, 0.81, 0.80, 0.82, 0.78, 0.77]  # Continuous relevance score

print(len(ranking_pred), len(sex), len(y_true))

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

# %%
# The new plots show that the exposure and exposure/utility ratio are now much more equal.
