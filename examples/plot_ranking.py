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
# This notebook shows how to apply functionalities of :mod:`fairlearn.metrics` to ranking problems.
# We showcase the example "Fairly Allocating Economic Opportunity" from the paper
# "Fairness of Exposure in Ranking" by Singh and Joachims (2018).
# The example demonstrates how small differences in item relevance can lead to large differences
# in exposure. Differences in exposure can be harmful not only because you are ranking individuals
# directly but individuals can also be indirect affected by a ranking, think about books, music or
# products in online retail.
#
# We reproduce the example of the paper, which for simplicity reasons uses a binary gender category
# as sensitive attribute. The metric however also works for multi-class sensitive attributes.
#
# Consider a web-service that connects employers ("users") to potential employees ("items").
# The web-service uses a ranking-based system to present a set of 6 applicants of which 3 are male
# and 3 are female. Male applicants have relevance of 0.80, 0.79, 0.78 respectively for the
# employer, while female applicants have relevance of 0.77, 0.76, 0.75.
# In this setting a relevance of 0.75 is defined as, 75% of all employers issuing the query
# considered the applicant relevant.
#
# The Probability Ranking Principle suggests to rank the applicants in decreasing order of
# relevance. What does this mean for the exposure between the two groups?
#
# NOTE: The used data should raise questions about
# :ref:`construct validity <fairness_in_machine_learning.construct-validity>` , since we
# should consider whether the "relevance" that is in the data is a good measurement of the actual
# relevance. Hindsight bias is also a concern, since how can you know upfront if an applicant
# will be a successful employee in the future? These concerns are important, but out of scope for
# this use case example.


ranking_pred = [1, 2, 3, 4, 5, 6]  # ranking
gender = ['Male', 'Male', 'Male', 'Female', 'Female', 'Female']
y_true = [0.82, 0.81, 0.80, 0.79, 0.78, 0.77]

# %%
# Here we define what metrics we want to analyze.
#
# - The :func:`fairlearn.metrics.exposure` metric measures the average exposure of a group of
#   items, based on their position in the ranking. Exposure is the value that we assign to every
#   place in the ranking, calculated by a
#   standard exposure drop-off of :math:`1/log_2(1+j)` as used in Discounted Cumulative Gain
#   :ref:`(DCG) <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.dcg_score.html>`
#   , to account for position bias. If there are big differences in exposure this could be an
#   indication of allocation harm, i.e. males are on average ranked way higher than females by the
#   web-service.
#
# - The :func:`fairlearn.metrics.utility` metric indicates the average "ground-truth" relevance of
#   a group.
#
# - The :func:`exposure_utility_ratio` metric computes the average exposure of a group, divided by
#   its utility (i.e., average relevance). Differences between groups indicate that the exposure of
#   some groups is not proportional to their ground-truth utility, which can be seen as a measure
#   of quality-of-service harm.
#
# We can compare ranking metrics across groups using :class:`fairlearn.metrics.MetricFrame`.

metrics = {
    'exposure (allocation harm)': exposure,
    'average utility': utility,
    'exposure/utility (quality-of-service)': exposure_utility_ratio
}

mf = MetricFrame(metrics=metrics,
                 y_true=y_true,
                 y_pred=ranking_pred,
                 sensitive_features={'gender': gender})

# Customize the plot
mf.by_group.plot(
    kind="bar",
    subplots=True,
    layout=[1, 3],
    legend=False,
    figsize=(12, 4)
)

# %%
# We can compute the minimum ratio between groups using
# :meth:`MetricFrame.ratio(method='between_groups')`. The closer the ratio is to 1 the more fair
# the ranking is.
mf.ratio()

# %%
# The first plot shows that the web-service that men get significantly more exposure than women.
# Although the second plot shows that the average utility of women is comparable to men.
# Therefor we can say that the ranking contains quality-of-service harm against women, since the
# exposure/utility ratio is not equal (plot 3)

# %%
# How can we fix this? A simple solution is to rerank the items, in such a way that females get
# more exposure and males get less exposure. For example we can swap the top male with the top
# female applicant and remeasure the quality-of-service harm.

ranking_pred = [1, 2, 3, 4, 5, 6]  # ranking
gender = ['Female', 'Male', 'Male', 'Male', 'Female', 'Female']
y_true = [0.79, 0.81, 0.80, 0.82, 0.78, 0.77]  # Continuous relevance score

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
                 sensitive_features={'gender': gender})

# Customize the plot
mf.by_group.plot(
    kind="bar",
    subplots=True,
    layout=[1, 3],
    legend=False,
    figsize=(12, 4)
)

mf.ratio()

# %%
# The new plots show that the exposure and exposure/utility ratio are now much more equal. The
# difference in exposure allocation is much smaller and the quality-of-service is better
# in proportion to the group's average utility.
