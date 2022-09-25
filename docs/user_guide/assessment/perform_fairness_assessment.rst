.. _perform_fairness_assessment:

Performing a Fairness Assessment
================================

.. currentmodule:: fairlearn.metrics

The goal of fairness assessment is to answer the question: Which groups of 
people may be disproportionately negatively impacted by an AI system and in 
what ways?

The steps of the assessment are as follows:

1. Identify types of harms 
2. Identify the groups that might be harmed 
3. Quantify harms 
4. Compare quantified harms across the groups 

We next examine these four steps in more detail.

Identify types of harms
-----------------------

See :ref:`types_of_harms` for a guide to types of fairness-related harms. 
For example, in a system for screening job applications, qualified candidates 
that are automatically rejected experience an allocation harm. In a 
speech-to-text transcription system, disparities in word error rates for 
different groups may result in harms due to differences in the quality of service.
Note that one system can lead to multiple harms, and different types of 
harms are not mutually exclusive. For more information, review 
Fairlearn's `2021 SciPy tutorial <https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/overview.pdf>`_.

Identify the groups that might be harmed
----------------------------------------

In most applications, we consider demographic groups including historically 
marginalized groups (e.g., based on gender, race, ethnicity). We should also 
consider groups that are relevant to a particular use case or deployment context. For example, for 
speech-to-text transcription, this might include groups who speak a regional dialect or people who are a  
native or a non-native speaker.

It is also important to consider group intersections, for example, in addition
to considering groups according to gender and groups according to race, it is 
also important to consider their intersections (e.g., Black women, Latinx 
nonbinary people, etc.). :footcite:cts:`crenshaw1991intersectionality`
offers a thorough background on the topic of intersectionality.
See :ref:`this section <assessment_intersecting_groups>` of our user guide for
details of how Fairlearn can compute metrics for intersections.


.. note::

    We have assumed that every sensitive feature is representable by a
    discrete variable.
    This is not always the case: for example, the melanin content of a
    person's skin (important for tasks such as facial recognition) will
    not be taken from a small number of fixed values.
    Features like this have to be binned, and the choice of bins
    could obscure fairness issues.

.. _assessment_quantify_harms:

Quantify harms
--------------

Define metrics that quantify harms or benefits:

* In a job screening scenario, we need to quantify the number of candidates
  that are classified as "negative" (not recommended for the job), but whose
  true label is "positive" (they are "qualified"). One possible metric is
  the false negative rate: fraction of qualified candidates that are
  screened out. Note that before we attempt to classify candidates, we need
  to determine the construct validity of the "qualified" status; more
  information on construct validity can be found in :ref:`construct_validity`

* For a speech-to-text application, the harm could be measured by disparities
  in the word error rate for different group, measured by the number of
  mistakes in a transcript divided by the overall number of words.

Note that in some cases, the outcome we seek to measure is not 
directly available. 
Occasionally, another variable in our dataset provides a close 
approximation to the phenomenon we seek to measure. 
In these cases, we might choose to use that closely related variable, 
often called a "proxy", to stand in for the missing variable. 
For example, suppose that in the job screening scenario, 
we have data on whether the candidate passes the first two stages, 
but not if they are ultimately recommended for the job. 

As an alternative to the unobserved final recommendation, we could
therefore measure the harm using the proxy variable indicating whether
the candidate passes the first stage of the screen.
If you choose to use a proxy variable to 
represent the harm, check the proxy variable regularly to ensure it 
remains useful over time. Our section on
:ref:`construct validity <construct_validity>`
describes how to determine whether a  
proxy variable measures the intended construct in a meaningful 
and useful way. It is important to ensure that the proxy is suitable 
for the social context of the problem you seek to solve. 
In particular, be careful of falling into one of the
:ref:`abstraction traps <abstraction_traps>`.

The centerpiece of fairness assessment in Fairlearn are disaggregated metrics, 
which are metrics evaluated on slices of data. For example, to measure harms due to 
errors, we would begin by evaluating the errors on each slice of the data that 
corresponds to a group. If some of the groups are seeing much larger errors 
than other groups, we would flag this as a fairness harm.

Fairlearn provides the :class:`fairlearn.metrics.MetricFrame` class to help
with this quantification.
Suppose we have some 'true' values, some predictions from a model, and also
a sensitive feature recorded for each.
The sensitive feature, denoted by :code:`sf_data`, can take on one of
three values:

.. doctest:: assessment_metrics

    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> sf_data = ['b', 'b', 'a', 'b', 'b', 'c', 'c', 'c', 'a',
    ...            'a', 'c', 'a', 'b', 'c', 'c', 'b', 'c', 'c']


Now, suppose we have determined that the metrics we are interested in are the
selection rate (:func:`selection_rate`), recall (a.k.a. true positive rate
:func:`sklearn.metrics.recall_score`) and false positive rate
(:func:`false_positive_rate`).
For completeness (and to help identify subgroups for which random noise might be
significant), we should also include the counts (:func:`count`).
We can use :class:`MetricFrame` to evaluate these metrics on our data:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', 20)
    >>> pd.set_option('display.width', 80)
    >>> from fairlearn.metrics import MetricFrame
    >>> from fairlearn.metrics import count, \
    ...                               false_positive_rate, \
    ...                               selection_rate
    >>> from sklearn.metrics import recall_score
    >>> # Construct a function dictionary
    >>> my_metrics = {
    ...     'tpr' : recall_score,
    ...     'fpr' : false_positive_rate,
    ...     'sel' : selection_rate,
    ...     'count' : count
    ... }
    >>> # Construct a MetricFrame
    >>> mf = MetricFrame(
    ...     metrics=my_metrics,
    ...     y_true=y_true,
    ...     y_pred=y_pred,
    ...     sensitive_features=sf_data
    ... )

We can now interrogate this :class:`MetricFrame` to find the values for
our chosen metrics.
First, the metrics evaluated on the entire dataset (disregarding the
sensitive feature), accessed via the :attr:`MetricFrame.overall`
property:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.overall
    tpr       0.500000
    fpr       0.666667
    sel       0.555556
    count    18.000000
    dtype: float64

Next, we can see the metrics evaluated on each of the groups identified by
the :code:`sf_data` column.
These are accessed through the :attr:`MetricFrame.by_group` property:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.by_group
                         tpr       fpr   sel  count
    sensitive_feature_0
    a                    0.5  1.000000  0.75    4.0
    b                    0.6  0.000000  0.50    6.0
    c                    0.4  0.666667  0.50    8.0

All of these values can be checked against the original arrays above.

.. _assessment_compare_harms:

Compare quantified harms across the groups
------------------------------------------

To summarize the disparities in errors (or other metrics), we may want to 
report quantities such as the difference or ratio of the metric values between 
the best and the worst slice. In settings where the goal is to guarantee 
certain minimum quality of service across all groups (such as speech recognition), 
it is also meaningful to report the worst performance across all considered groups.

The :class:`MetricFrame` class provides several methods for comparing
the computed metrics.
For example, the :meth:`MetricFrame.group_min` and :meth:`MetricFrame.group_max`
methods show the smallest and largest values for each metric:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.group_min()
    tpr      0.4
    fpr      0.0
    sel      0.5
    count    4.0
    dtype: object
    >>> mf.group_max()
    tpr       0.6
    fpr       1.0
    sel      0.75
    count     8.0
    dtype: object

We can also compute differences and ratios between groups for all of the
metrics.
These are available via the :meth:`MetricFrame.difference` and
:meth:`MetricFrame.ratio` methods respectively.
The absolute difference will always be returned, and the ratios will be chosen
to be less than one.
By default, the computations are done between the maximum and minimum
values for the groups:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.difference()
    tpr      0.20
    fpr      1.00
    sel      0.25
    count    4.00
    dtype: float64
    >>> mf.ratio()
    tpr      0.666667
    fpr      0.000000
    sel      0.666667
    count    0.500000
    dtype: float64

However, the differences and ratios can also be computed relative to the
overall values for the data:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.difference(method='to_overall')
    tpr       0.100000
    fpr       0.666667
    sel       0.194444
    count    14.000000
    dtype: float64
    >>> mf.ratio(method='to_overall')
    tpr      0.800000
    fpr      0.000000
    sel      0.740741
    count    0.222222
    dtype: float64

In every case, the *largest* difference and *smallest* ratio are returned.