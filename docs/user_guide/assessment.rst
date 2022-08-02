Assessment
==========

.. currentmodule:: fairlearn.metrics

In this section, we will describe the steps involved in performing a fairness
assessment, and show how :class:`MetricFrame` can be
used to assist in this process.

Performing a Fairness Assessment
--------------------------------

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
^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`types_of_harms` for a guide to types of fairness-related harms. 
For example, in a system for screening job applications, qualified candidates 
that are automatically rejected experience an allocation harm. In a 
speech-to-text transcription system, disparities in word error rates for 
different groups may result in harms due to differences in the quality of service.
Note that one system can lead to multiple harms, and different types of 
harms are not mutually exclusive. For more information, review 
Fairlearn's `2021 SciPy tutorial <https://github.com/fairlearn/talks/blob/main/2021_scipy_tutorial/overview.pdf>`_.

Identify the groups that might be harmed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

One important thing to note here: we have assumed that every sensitive
feature is representable by a discrete variable.
This is not always the case: for example, the melanin content of a
person's skin (important for tasks such as facial recognition) will
not be taken from a small number of fixed values.
For now, features like this have to be binned.
To reduce the risk of harms being blurred out by the binning,
analyses should be repeated with the number of bins changed and/or
the bin edges moved.


Quantify harms
^^^^^^^^^^^^^^

Define metrics that quantify harms or benefits:

* In a job screening scenario, we need to quantify the number of candidates that are classified as "negative" (not recommended for the job), but whose true label is "positive" (they are "qualified"). One possible metric is the false negative rate: fraction of qualified candidates that are screened out. Note that before we attempt to classify candidates, we need to determine the construct validity of the "qualified" status; more information on construct validity can be found in :ref:`construct_validity`

* For a speech-to-text application, the harm could be measured by disparities in the word error rate for different group, measured by the number of mistakes in a transcript divided by the overall number of words.

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
remains useful over time. Our section on :ref:`construct_validity`
describes how to determine whether a  
proxy variable measures the intended construct in a meaningful 
and useful way. It is important to ensure that the proxy is suitable 
for the social context of the problem you seek to solve. 
In particular, be careful of falling into one of the :ref:abstraction_traps. 

Compare quantified harms across the groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The centerpiece of fairness assessment in Fairlearn are disaggregated metrics, 
which are metrics evaluated on slices of data. For example, to measure harms due to 
errors, we would begin by evaluating the errors on each slice of the data that 
corresponds to a group. If some of the groups are seeing much larger errors 
than other groups, we would flag this as a fairness harm.

To summarize the disparities in errors (or other metrics), we may want to 
report quantities such as the difference or ratio of the metric values between 
the best and the worst slice. In settings where the goal is to guarantee 
certain minimum quality of service across all groups (such as speech recognition), 
it is also meaningful to report the worst performance across all considered groups.

For example, when comparing the false negative rate across groups defined by race, 
we may summarize our findings with a table. Note that the these statistics must 
be drawn from a large enough sample size to draw meaningful conclusions. 

.. list-table::
   :header-rows: 1
   :widths: 7 30 30
   :stub-columns: 1

   *  - 
      - false negative rate (FNR)
      - sample size

   *  - AfricanAmerican
      - 0.43
      - 126

   *  - Caucasian
      - 0.44
      - 620

   *  - Other
      - 0.52
      - 200

   *  - Unknown
      - 0.67
      - 60

   *  - largest difference
      - 0.24 (best is 0.0)
      - N/A

   *  - smallest ratio
      - 0.64 (best is 1.0)
      - N/A

   *  - maximum (worst-case) FNR
      - 0.67
      - N/A

Metrics & Disaggregated metrics
-------------------------------

Now that we have defined the steps of a fairness analysis, we can look at how
the :py:mod:`fairlearn.metrics` module can help.
Obviously, it can only help with the last two steps - quantifying harms and
comparing those harms between groups.

In the mathematical definitions below, :math:`X` denotes a feature vector 
used for predictions, :math:`A` will be a single sensitive feature (such as age 
or race), and :math:`Y` will be the true label.
Fairness metrics are phrased in terms of expectations with respect to the
distribution over :math:`(X,A,Y)`.
Note that :math:`X` and :math:`A` may or may not share columns, dependent on
whether the model is allowed to 'see' the sensitive features.
When we need to refer to particular values, we will use lower case letters;
since we are going to be comparing between groups identified by the
sensitive feature, :math:`\forall a \in A` will be appearing regularly to
indicate that a property holds for all identified groups.

At their simplest, metrics take a vector of 'true objects' :math:`Y_{true}` (from
the input data) and 'predicted objects' :math:`Y_{pred}` (by applying the model
to the input data), and use these to compute a measure.
*We are being deliberately vague* as to what constitutes a 'true object' or a
'measure' at this point.

For the sake of concreteness, we shall assume for the time being that we
are working on a binary classification problem, where the two classes have
labels of 0 and 1.
We can then use a number of conventional metrics to illustrate the basic
usage of :class:`MetricFrame`.
However, we will return to more complex cases later in this discussion.
For now, consider the *recall* or *true positive rate*, which is given by

.. math::

   P( Y_{pred}=1 \given Y_{true}=1 )

That is, a measure of whether the model finds all the positive cases in the
input data. The `scikit-learn` package implements this in
:py:func:`sklearn.metrics.recall_score`.

Suppose we have the following data we can see that the prediction is `1` in five
of the ten cases where the true value is `1`, so we expect the recall to be 0.5:

.. doctest:: assessment_metrics

    >>> import sklearn.metrics as skm
    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> skm.recall_score(y_true, y_pred)
    0.5

In a typical fairness assessment, each row of input data will have an associated
group label :math:`a \in A`, and we will want to know how the metric behaves
for each group :math:`a`.
To help with this, Fairlearn provides a class that takes
an existing metric function, like 
:func:`sklearn.metrics.roc_auc_score` or :func:`fairlearn.metrics.false_positive_rate`, 
and applies it to each group within a set of data.

This data structure, :class:`fairlearn.metrics.MetricFrame`, enables evaluation 
of disaggregated metrics. In its simplest form :class:`fairlearn.metrics.MetricFrame` 
takes four arguments:

* metric_function with signature :code:`metric_function(y_true, y_pred)`

* y_true: array of labels

* y_pred: array of predictions

* sensitive_features: array of sensitive feature values

The code chunk below displays a case where in addition to the :math:`Y_{true}` 
and :math:`Y_{pred}` above, the dataset also contains the following set of 
labels, denoted by the "group_membership_data" column:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> import numpy as np
    >>> import pandas as pd
    >>> group_membership_data = ['b', 'b', 'a', 'b', 'b', 'c', 'c', 'c', 'a',
    ...                          'a', 'c', 'a', 'b', 'c', 'c', 'b', 'c', 'c']
    >>> pd.set_option('display.max_columns', 20)
    >>> pd.set_option('display.width', 80)
    >>> pd.DataFrame({ 'y_true': y_true,
    ...                'y_pred': y_pred,
    ...                'group_membership_data': group_membership_data})
        y_true  y_pred group_membership_data
    0        0       0                     b
    1        1       0                     b
    2        1       1                     a
    3        1       0                     b
    4        1       1                     b
    5        0       1                     c
    6        1       1                     c
    7        0       0                     c
    8        1       0                     a
    9        0       1                     a
    10       0       1                     c
    11       0       1                     a
    12       1       1                     b
    13       1       0                     c
    14       1       0                     c
    15       1       1                     b
    16       1       1                     c
    17       1       0                     c
    <BLANKLINE>

We then calculate a metric which shows the subgroups:

.. doctest:: assessment_metrics

    >>> from fairlearn.metrics import MetricFrame
    >>> grouped_metric = MetricFrame(metrics=skm.recall_score,
    ...                              y_true=y_true,
    ...                              y_pred=y_pred,
    ...                              sensitive_features=group_membership_data)
    >>> print("Overall recall = ", grouped_metric.overall)
    Overall recall =  0.5
    >>> print("recall by groups = ", grouped_metric.by_group.to_dict())
    recall by groups =  {'a': 0.5, 'b': 0.6, 'c': 0.4}

The disaggregated metrics are stored in a :class:`pandas.Series` 
:code:`grouped_metric.by_group`.
Note that the :meth:`MetricFrame.overall` property has the value of 0.5, as
we obtained above by applying :py:func:`sklearn.metrics.recall_score` to the
whole dataset.
The :meth:`MetricFrame.by_group` property (which we turned into a dictionary
for display purposes) can be checked against the table above.

In addition to these basic scores, Fairlearn provides
convenience functions to recover the maximum and minimum values of the metric
across groups and also the difference and ratio between the maximum and minimum:

.. doctest:: assessment_metrics

    >>> print("min recall over groups = ", grouped_metric.group_min())
    min recall over groups =  0.4
    >>> print("max recall over groups = ", grouped_metric.group_max())
    max recall over groups =  0.6
    >>> print("difference in recall = {:3f}".format(grouped_metric.difference(method='between_groups')))
    difference in recall = 0.200000
    >>> print("ratio in recall = {:3f}".format(grouped_metric.ratio(method='between_groups')))    
    ratio in recall = 0.666667

The difference and ratio calculations can also be made relative to the
overall value, rather than the largest and smallest values of
:meth:`MetricFrame.by_group`.
We simply change the value of the :code:`method` argument:

.. doctest:: assessment_metrics

    >>> print('{:3f}'.format(grouped_metric.difference(method='to_overall')))
    0.100000
    >>> print('{:3f}'.format(grouped_metric.ratio(method='to_overall')))
    0.800000


Common fairness metrics
-----------------------
In the sections below, we review the most common fairness metrics, as well
as their underlying assumptions and suggestions for use. Each metric requires
that some aspects of the predictor behavior be comparable across groups.

.. _demographic_parity:

Demographic parity
^^^^^^^^^^^^^^^^^^
Demographic parity is a fairness metric whose goal is to ensure a machine 
learning model's predictions are independent of membership in a sensitive 
group. In other words, demographic parity is achieved when the probability 
of a certain prediction is not dependent on sensitive group membership. In 
the binary classification scenario, demographic parity refers to equal 
selection rates across groups. For example, in the context of a resume 
screening model, equal selection would mean that the proportion of 
applicants selected for a job interview should be equal across groups.

We mathematically define demographic parity using the following 
set of equations.
A classifier :math:`h` satisfies demographic parity under a distribution 
over :math:`(X, A, Y)` if its prediction :math:`h(X)` is statistically
independent of the sensitive feature :math:`A`.
:footcite:cts:`agarwal2018reductions` show that this is equivalent to
:math:`\E[h(X) \given A=a] = \E[h(X)] \quad \forall a`.

In the case of regression, a predictor :math:`f` satisfies demographic parity
under a distribution over :math:`(X, A, Y)` if :math:`f(X)` is independent
of the sensitive feature :math:`A`.
:footcite:cts:`agarwal2019fair` show that this is equivalent to
:math:`\P[f(X) \geq z \given A=a] = \P[f(X) \geq z] \quad \forall a, z`.
Another way to think of demographic parity in a 
regression scenario is to compare the average predicted value across groups.
Note that in the Fairlearn API, :func:`fairlearn.metrics.demographic_parity_difference` 
is only defined for classification. 

.. note::
   Demographic parity is also sometimes referred to as *independence*, *group fairness*, *statistical parity*, and *disparate impact*.

Failing to achieve demographic parity could generate allocation harms. 
Allocation harms occur when AI systems allocate 
opportunities, resources, or information differently across different 
groups (for example, an AI hiring system that is more likely to advance resumes 
of male applicants than resumes of female applicants regardless of qualification). 
Demographic parity can be used to assess the extent of allocation harms because it 
reflects an assumption that resources should be allocated proportionally 
across groups. Of the metrics described in this section, it can be the easiest 
to implement. However, operationalizing fairness using demographic parity 
rests on a few assumptions: that either the dataset is not a good representation  
of what the world actually looks like (e.g., a resume assessment system that is 
more likely to filter out qualified female applicants due to an organizational 
bias towards male applicants, regardless of skill level), or that the dataset 
is an accurate representation of the phenomena being modeled, but the 
phenomena itself is unjust (e.g., consider the case of predictive policing, 
where a system created to predict crime rates may correctly predict higher crime 
rates for certain areas, but simultaneously fail to consider that those higher 
rates may be caused by disproportionate policing and overcriminimalization of those areas). 
In reality, these assumptions may not be the true. The
dataset might be an accurate representation of the phenomena itself, 
or the phenomena being modeled may not be unjust. 
If either assumption is not true, then demographic parity may not provide 
a meaningful or useful measurement of the fairness of a model's predictions. 

Fairness metrics like demographic parity can also be used as optimization 
constraints during the machine learning model training process. However, 
demographic parity may not be well-suited for this purpose because 
it does not place requirements on the exact distribution of predictions with 
respect to other important variables. To understand this concept further, 
consider an example from the Fairness in Machine Learning textbook 
by :footcite:cts:`barocas2019fairness`:

    "However, decisions based on a classifier that satisfies independence can 
    have undesirable properties (and similar arguments apply to other 
    statistical critiera). Here is one way in which this can happen, 
    which is easiest to illustrate if we imagine a callous or ill-intentioned 
    decision maker. Imagine a company that in *group A* hires diligently 
    selected applicants at some rate p>0. In *group B*, the company 
    hires carelessly selected applicants at the same rate p. Even though 
    the acceptance rates in both groups are identical, it is far more likely 
    that unqualified applicants are selected in one group than in the other. 
    As a result, it will appear in hindsight that members of *group B* 
    performed worse than members of *group A*, thus establishing a negative 
    track record for group B."

It's also worth considering whether the assumptions underlying demographic
parity maintain construct validity (see :ref:`construct_validity`). 
Construct validity is a concept in the social sciences that assesses the 
extent to which the ways we choose to measure abstract 
phenomena are valid. For demographic parity, one relevant question would be 
whether demographic parity meets the criteria for establishing "fairness", 
itself an unobservable theoretical construct. Further, it's important 
to ask whether satisfying demographic parity actually brings us closer 
to the world we'd like to see. 


.. _conditional_group_fairness:

In some cases, we may observe a trend in data from multiple demographic groups, 
but that trend may disappear or reverse when groups are combined. Known as  
`Simpson's Paradox <https://en.wikipedia.org/wiki/Simpson%27s_paradox>`_, this 
outcome may appear when observing disparate outcomes across groups. A 
famous example of Simpson's Paradox is a study of 1973 graduate school 
admissions to the University of California, Berkley :footcite:ps:`bickel1975biasinadmissions`. 
The study showed that when observing admissions by gender, men applying were 
more likely than women to be accepted. However, drilling down into admissions 
by department revealed that women tended to apply to departments with more 
competitive admissions requirements, whereas men tended to apply to less 
competitive departments. The more granular analysis showed only four out of 
85 departments exhibited bias against women, and six departments exhibited 
bias towards men. In general, the data indicated departments exhibited a bias 
in favor of minority-gendered applicants, which is opposite from the trend 
observed in the aggregate data.

This phenomenon is important to fairness evaluation because metrics like 
demographic parity may be different when calculated at an aggregate level and  
within more granular categories. In the case of demographic parity, we might 
need to review 
:math:`\E[h(X) \given A=a, D=d] = \E[h(X) \given D=d] \quad \forall a` 
where :math:`D` represents the feature(s) within :math:`X`` across which
members of the groups within :math:`A` are distributed. 
Demographic parity would then require that the prediction of the target  
variable is statistically independent of sensitive attributes conditional 
on D. Simply aggregating outcomes across high-level categories can be 
misleading when the data can be further disaggregated. 
It's important to review metrics across these more granular categories, 
if they exist, to verify that disparate outcomes persist across all levels 
of aggregation. 

However, more granular categories generally contain smaller sample sizes, and 
it can be more difficult to establish that trends seen in very small 
samples are not due to random chance. 
We also recommend watching out for the `multiple comparisons problem <https://stats.libretexts.org/Bookshelves/Applied_Statistics/Book%3A_Biological_Statistics_(McDonald)/06%3A_Multiple_Tests/6.01%3A_Multiple_Comparisons>`_, 
which states that the more statistical inferences are made, the 
more erroneous those inferences will become. For example, in the case 
of evaluating fairness metrics on multiple groups, as we break the 
groups down into more granular categories and evaluate those smaller 
groups, it will become more likely that these subgroups will 
differ enough to fail one of the metrics. For dealing with the multiple 
comparisons problem, we recommend investigating `statistical techniques <https://www.statology.org/bonferroni-correction/>`_ 
meant to correct the errors produced by individual statistical tests. 

Fairlearn provides the :func:`demographic_parity_difference` and
:func:`demographic_parity_ratio` functions for computing demographic
parity measures for binary classification data, both of which return
a scalar result
The first reports the absolute difference between the highest and
lowest selection rates :math:`a \in A` so a result of 0 indicates
that demographic parity has been achieved (this does *not* automatically
mean that the classifier is fair!).
The second reports the ratio of the lowest and highest selection rates,
so a result of 1 means there is demographic parity.


.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import demographic_parity_difference
    >>> print(demographic_parity_difference(y_true,
    ...                                     y_pred,
    ...                                     sensitive_features=group_membership_data))
    0.25
    >>> from fairlearn.metrics import demographic_parity_ratio
    >>> print(demographic_parity_ratio(y_true,
    ...                                y_pred,
    ...                                sensitive_features=group_membership_data))
    0.66666...


.. _equalized_odds:

Equalized odds
^^^^^^^^^^^^^^
The goal of the equalized odds fairness metric is to ensure a machine 
learning model performs equally well for different groups. It is stricter 
than demographic parity because it requires that the machine learning 
model's predictions are not only independent of sensitive group membership, 
but that groups have the same false positive rates and and true positive 
rates. This distinction is important because a model could achieve 
demographic parity (i.e., its predictions could be independent of 
sensitive group membership), but still generate more false positive 
predictions for one group versus others. Equalized odds does not create 
the selection issue discussed in the demographic parity section above. 
For example, in the hiring scenario where the goal is to choose applicants 
from *group A* and *group B*, ensuring the model performs equally well at 
choosing applicants from *group A* and *group B* can circumvent the issue of 
the model optimizing by selecting applicants from one group at random.

We mathematically define equalized odds using the following 
set of equations. A classifier :math:`h` satisfies equalized 
odds under a distribution over :math:`(X, A, Y)` if its 
prediction :math:`h(X)` is
conditionally independent of the sensitive feature :math:`A` given the label
:math:`Y`.
:footcite:cts:`agarwal2018reductions` show that this is equivalent to
:math:`\E[h(X) \given A=a, Y=y] = \E[h(X) \given Y=y] \quad \forall a, y`.
Equalized odds requires that the true 
positive rate, :math:`\P(h(X)=1 | Y=1`, and the false positive rate, 
:math:`\P(h(X)=1 | Y=0`, be equal across groups. 

The inclusion of false positive rates acknowledges that different groups 
experience different costs from misclassification. For example, in the case of 
a model predicting a negative outcome (e.g., probability of recidivating) 
that already disproportionately affects members of minority communities, 
false positive predictions reflect pre-existing disparities in outcomes 
across minority and majority groups. Equalized odds further enforces that the 
accuracy is equally high across all groups, punishing models that only 
perform well on majority groups.

If a machine learning model does not perform equally well for all groups, 
then it could generate allocation or quality-of-service harms.
Equalized odds can be used to diagnose both allocation harms as well as 
quality-of-service harms. Allocation harms are discussed in detail in the 
demographic parity section above. Quality-of-service harms occur when an 
AI system does not work as well for one group versus another (for example, 
facial recognition systems that are more likely to fail for dark-skinned 
individuals). For more information on AI harms, see :ref:`types_of_harms`. 

Equalized odds can be useful for diagnosing allocation harms 
because its goal is to ensure that a machine learning model works equally 
well for different groups. Another way to think about equalized odds is to 
contrast it with demographic parity. While demographic parity assesses the 
allocation of resources generally, equalized odds focuses on the allocation 
of resources that were actually distributed to 
members of that group (indicated by the positive target variable :math:`Y=1``). 
However, equalized odds makes the assumption 
that the target variable :math:`Y` is a good measurement of the phenomena 
being modeled, but that assumption may not hold if the measurement does not 
satisfy the requirements of construct validity.

Similar to the demographic parity case, Fairlearn provides
:func:`equalized_odds_difference` and :func:`equalized_odds_ratio`
to help with these calculations.
However, since equalized odds is based on both the true positive and
false positive rates, there is an extra step in order to return
a single scalar result.
For :func:`equalized_odds_difference`, we first calculate the
true positive rate difference and the true negative rate difference
separately.
We then return the larger of these two differences.
*Mutatis mutandis*, :func:`equalized_odds_ratio` works similarly.


.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import equalized_odds_difference
    >>> print(equalized_odds_difference(y_true,
    ...                                 y_pred,
    ...                                 sensitive_features=group_membership_data))
    1.0
    >>> from fairlearn.metrics import equalized_odds_ratio
    >>> print(equalized_odds_ratio(y_true,
    ...                            y_pred,
    ...                            sensitive_features=group_membership_data))
    0.0



.. _equal_opportunity:

Equal opportunity
^^^^^^^^^^^^^^^^^
Equal opportunity is a relaxed version of equalized odds that only considers
conditional expectations with respect to positive labels, i.e., :math:`Y=1`.
:footcite:p:`hardt2016equality`
Another way of thinking about this metric is 
requiring equal outcomes only within the subset of records belonging to the 
positive class. For example, in the hiring example, equal opportunity 
requires that the individuals who are actually hired have an equal opportunity 
of being hired in the first place. However, by not considering whether false 
positive rates are equivalent across groups, equal opportunity does not 
capture the costs of missclassification disparities.



.. _scalar_metric_results:

Scalar results from :code:`MetricFrame`
---------------------------------------

Higher level machine learning algorithms (such as hyperparameter tuners) often
make use of metric functions to guide their optimisations.
Such algorithms generally work with scalar results, so if we want the tuning
to be done on the basis of our fairness metrics, we need to perform aggregations
over the :class:`MetricFrame`.

We provide a convenience function, :func:`fairlearn.metrics.make_derived_metric`,
to generate scalar-producing metric functions based on the aggregation methods
mentioned above (:meth:`MetricFrame.group_min`, :meth:`MetricFrame.group_max`,
:meth:`MetricFrame.difference`, and :meth:`MetricFrame.ratio`).
This takes an underlying metric function, the name of the desired transformation, and
optionally a list of parameter names which should be treated as sample aligned parameters
(such as `sample_weight`).
The result is a function which builds the :class:`MetricFrame` internally and performs
the requested aggregation. For example:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import make_derived_metric
    >>> recall_difference = make_derived_metric(metric=skm.recall_score,
    ...                                        transform='difference')
    >>> recall_difference(y_true, y_pred,
    ...                   sensitive_features=group_membership_data)
    0.19999...
    >>> MetricFrame(metrics=skm.recall_score,
    ...             y_true=y_true,
    ...             y_pred=y_pred,
    ...             sensitive_features=group_membership_data).difference()
    0.19999...

We use :func:`fairlearn.metrics.make_derived_metric` to manufacture a number
of such functions which will be commonly used:

=============================================== ================= ================= ================== =============
Base metric                                     :code:`group_min` :code:`group_max` :code:`difference` :code:`ratio`
=============================================== ================= ================= ================== =============
:func:`.false_negative_rate`                    .                 .                 Y                  Y
:func:`.false_positive_rate`                    .                 .                 Y                  Y
:func:`.selection_rate`                         .                 .                 Y                  Y
:func:`.true_negative_rate`                     .                 .                 Y                  Y
:func:`.true_positive_rate`                     .                 .                 Y                  Y
:func:`sklearn.metrics.accuracy_score`          Y                 .                 Y                  Y
:func:`sklearn.metrics.balanced_accuracy_score` Y                 .                 .                  .
:func:`sklearn.metrics.f1_score`                Y                 .                 .                  .
:func:`sklearn.metrics.log_loss`                .                 Y                 .                  .
:func:`sklearn.metrics.mean_absolute_error`     .                 Y                 .                  .
:func:`sklearn.metrics.mean_squared_error`      .                 Y                 .                  .
:func:`sklearn.metrics.precision_score`         Y                 .                 .                  .
:func:`sklearn.metrics.r2_score`                Y                 .                 .                  .
:func:`sklearn.metrics.recall_score`            Y                 .                 .                  .
:func:`sklearn.metrics.roc_auc_score`           Y                 .                 .                  .
:func:`sklearn.metrics.zero_one_loss`           .                 Y                 Y                  Y
=============================================== ================= ================= ================== =============

The names of the generated functions are of the form
:code:`fairlearn.metrics.<base_metric>_<transformation>`.
For example :code:`fairlearn.metrics.accuracy_score_difference` and
:code:`fairlearn.metrics.precision_score_group_min`.


Multiple metrics in a single :code:`MetricFrame`
------------------------------------------------

A single instance of :class:`fairlearn.metrics.MetricFrame` can evaluate multiple
metrics simultaneously by providing the `metrics` argument with a 
dictionary of desired metrics. The disaggregated metrics are then stored in a 
pandas DataFrame. Note that :class:`pandas.DataFrame` can 
be used to show each group's size:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import count
    >>> multi_metric = MetricFrame({'precision':skm.precision_score,
    ...                             'recall':skm.recall_score,
    ...                             'count': count},
    ...                             y_true, y_pred,
    ...                             sensitive_features=group_membership_data)
    >>> multi_metric.overall
    precision     0.6
    recall        0.5
    count        18.0
    dtype: float64
    >>> multi_metric.by_group
                         precision  recall  count
    sensitive_feature_0
    a                     0.333333     0.5    4.0
    b                     1.000000     0.6    6.0
    c                     0.500000     0.4    8.0



Intersecting Groups
-------------------

The :class:`MetricFrame` class supports intersectional fairness in two ways:
multiple sensitive features, and control features.
Both of these can be used simultaneously.
One important point to bear in mind when performing an intersectional analysis
is that some of the intersections may have very few members (or even be empty).
This can affect the confidence interval associated with the computed metrics.

Multiple Sensitive Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Multiple sensitive features can be specified when the :class:`MetricFrame`
is constructed.
The ``by_groups`` property then holds the intersections of these groups:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> g_2 = [ 8,6,8,8,8,8,6,6,6,8,6,6,6,6,8,6,6,6 ]
    >>> s_f_frame = pd.DataFrame(np.stack([group_membership_data, g_2], axis=1),
    ...                          columns=['SF 0', 'SF 1'])
    >>> metric_2sf = MetricFrame(metrics=skm.recall_score,
    ...                          y_true=y_true,
    ...                          y_pred=y_pred,
    ...                          sensitive_features=s_f_frame)
    >>> metric_2sf.overall  # Same as before
    0.5
    >>> metric_2sf.by_group
    SF 0  SF 1
    a     6       0.000000
          8       1.000000
    b     6       0.666667
          8       0.500000
    c     6       0.500000
          8       0.000000
    Name: recall_score, dtype: float64

If a particular intersection of the sensitive features had no members, then
the metric would be shown as :code:`NaN` for that intersection.
Multiple metrics can also be computed at the same time:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> metric_2sf_multi = MetricFrame(
    ...     metrics={'precision':skm.precision_score,
    ...              'recall':skm.recall_score,
    ...              'count': count},
    ...     y_true=y_true,
    ...     y_pred=y_pred,
    ...     sensitive_features=s_f_frame
    ... )
    >>> metric_2sf_multi.overall
    precision     0.6
    recall        0.5
    count        18.0
    dtype: float64
    >>> metric_2sf_multi.by_group
               precision    recall  count
    SF 0 SF 1
    a    6      0.000000  0.000000    2.0
         8      0.500000  1.000000    2.0
    b    6      1.000000  0.666667    3.0
         8      1.000000  0.500000    3.0
    c    6      0.666667  0.500000    6.0
         8      0.000000  0.000000    2.0


Control Features
^^^^^^^^^^^^^^^^

Control features (sometimes called 'conditional' features) enable more detailed
fairness insights by providing a further means of splitting the data into
subgroups.
When the data are split into subgroups, control features (if provided) act
similarly to sensitive features.
However, the 'overall' value for the metric is now computed for each subgroup
of the control feature(s).
Similarly, the aggregation functions (such as :code:`MetricFrame.group_max`) are
performed for each subgroup in the conditional feature(s), rather than across
them (as happens with the sensitive features).

Control features are useful for cases where there is some expected variation with
a feature, so we need to compute disparities while controlling for that feature.
For example, in a loan scenario we would expect people of differing incomes to
be approved at different rates, but within each income band we would still
want to measure disparities between different sensitive features.
**However**, it should be borne in mind that due to historic discrimination, the
income band might be correlated with various sensitive features.
Because of this, control features should be used with particular caution.

The :class:`MetricFrame` constructor allows us to specify control features in
a manner similar to sensitive features, using a :code:`control_features=`
parameter:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> decision = [
    ...    0,0,0,1,1,0,1,1,0,1,
    ...    0,1,0,1,0,1,0,1,0,1,
    ...    0,1,1,0,1,1,1,1,1,0
    ... ]
    >>> prediction = [
    ...    1,1,0,1,1,0,1,0,1,0,
    ...    1,0,1,0,1,1,1,0,0,0,
    ...    1,1,1,0,0,1,1,0,0,1
    ... ]
    >>> control_feature = [
    ...    'H','L','H','L','H','L','L','H','H','L',
    ...    'L','H','H','L','L','H','L','L','H','H',
    ...    'L','H','L','L','H','H','L','L','H','L'
    ... ]
    >>> sensitive_feature = [
    ...    'A','B','B','C','C','B','A','A','B','A',
    ...    'C','B','C','A','C','C','B','B','C','A',
    ...    'B','B','C','A','B','A','B','B','A','A'
    ... ]
    >>> metric_c_f = MetricFrame(metrics=skm.accuracy_score,
    ...                          y_true=decision,
    ...                          y_pred=prediction,
    ...                          sensitive_features={'SF' : sensitive_feature},
    ...                          control_features={'CF' : control_feature})
    >>> # The 'overall' property is now split based on the control feature
    >>> metric_c_f.overall
    CF
    H    0.4285...
    L    0.375...
    Name: accuracy_score, dtype: float64
    >>> # The 'by_group' property looks similar to how it would if we had two sensitive features
    >>> metric_c_f.by_group
    CF  SF
    H   A     0.2...
        B     0.4...
        C     0.75...
    L   A     0.4...
        B     0.2857...
        C     0.5...
    Name: accuracy_score, dtype: float64

Note how the :attr:`MetricFrame.overall` property is stratified based on the
supplied control feature. The :attr:`MetricFrame.by_group` property allows
us to see disparities between the groups in the sensitive feature for each
group in the control feature.
When displayed like this, :attr:`MetricFrame.by_group` looks similar to
how it would if we had specified two sensitive features (although the
control features will always be at the top level of the hierarchy).

With the :class:`MetricFrame` computed, we can perform aggregations:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> # See the maximum accuracy for each value of the control feature
    >>> metric_c_f.group_max()
    CF
    H    0.75
    L    0.50
    Name: accuracy_score, dtype: float64
    >>> # See the maximum difference in accuracy for each value of the control feature
    >>> metric_c_f.difference(method='between_groups')
    CF
    H    0.55...
    L    0.2142...
    Name: accuracy_score, dtype: float64

In each case, rather than a single scalar, we receive one result for each
subgroup identified by the conditional feature. The call
:code:`metric_c_f.group_max()` call shows the maximum value of the metric across
the subgroups of the sensitive feature within each value of the control feature.
Similarly, :code:`metric_c_f.difference(method='between_groups')` call shows the
maximum difference between the subgroups of the sensitive feature within
each value of the control feature.
For more examples, please
see the :ref:`sphx_glr_auto_examples_plot_new_metrics.py` notebook in the
:ref:`examples`.

Finally, a :class:`MetricFrame` can use multiple control features, multiple
sensitive features and multiple metric functions simultaneously.


Extra Arguments to Metric functions
-----------------------------------

The metric functions supplied to :class:`MetricFrame` might require additional
arguments.
These fall into two categories: 'scalar' arguments (which affect the operation
of the metric function), and 'per-sample' arguments (such as sample weights).
Different approaches are required to use each of these.

Scalar Arguments
^^^^^^^^^^^^^^^^

We do not directly support scalar arguments for the metric functions.
If these are required, then use :func:`functools.partial` to prebind the
required arguments to the metric function:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> import functools
    >>> fbeta_06 = functools.partial(skm.fbeta_score, beta=0.6)
    >>> metric_beta = MetricFrame(metrics=fbeta_06,
    ...                           y_true=y_true,
    ...                           y_pred=y_pred,
    ...                           sensitive_features=group_membership_data)
    >>> metric_beta.overall
    0.56983...
    >>> metric_beta.by_group
    sensitive_feature_0
    a    0.365591
    b    0.850000
    c    0.468966
    Name: metric, dtype: float64


Per-Sample Arguments
^^^^^^^^^^^^^^^^^^^^

If there are per-sample arguments (such as sample weights), these can also be 
provided in a dictionary via the ``sample_params`` argument:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> s_w = [1, 2, 1, 3, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 2, 3, 1, 1]
    >>> s_p = { 'sample_weight':s_w }
    >>> weighted = MetricFrame(metrics=skm.recall_score,
    ...                        y_true=y_true,
    ...                        y_pred=y_pred,
    ...                        sensitive_features=pd.Series(group_membership_data, name='SF 0'),
    ...                        sample_params=s_p)
    >>> weighted.overall
    0.45...
    >>> weighted.by_group
    SF 0
    a    0.500000
    b    0.583333
    c    0.250000
    Name: recall_score, dtype: float64

If multiple metrics are being evaluated, then ``sample_params`` becomes a 
dictionary of dictionaries, with the first key corresponding matching that in 
the dictionary holding the desired underlying metric functions.
For example:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> s_w_2 = [3, 1, 2, 3, 2, 3, 1, 4, 1, 2, 3, 1, 2, 1, 4, 2, 2, 3]
    >>> metrics = {
    ...    'recall' : skm.recall_score,
    ...    'recall_weighted' : skm.recall_score,
    ...    'recall_weight_2' : skm.recall_score
    ... }
    >>> s_p = {
    ...     'recall_weighted' : { 'sample_weight':s_w },
    ...     'recall_weight_2' : { 'sample_weight':s_w_2 }
    ... }
    >>> weighted = MetricFrame(metrics=metrics,
    ...                        y_true=y_true,
    ...                        y_pred=y_pred,
    ...                        sensitive_features=pd.Series(group_membership_data, name='SF 0'),
    ...                        sample_params=s_p)
    >>> weighted.overall
    recall             0.500000
    recall_weighted    0.454545
    recall_weight_2    0.458333
    dtype: float64
    >>> weighted.by_group
          recall  recall_weighted  recall_weight_2
    SF 0
    a        0.5         0.500000         0.666667
    b        0.6         0.583333         0.600000
    c        0.4         0.250000         0.272727


More Complex Metrics
--------------------

So far, we have stuck to relatively simple cases, where the inputs are 1-D vectors of scalars,
and the metric functions return scalar values.
However, this need not be the case - we noted above that we were going to be vague as to the
contents of the input vectors and the return value of the metric function.

Metric Function Returns Non-Scalar
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Metric functions need not return a scalar value.
A straightforward example of this is the confusion matrix.
Such return values are fully supported by :class:`MetricFrame`:


.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> mf_conf = MetricFrame(
    ...    metrics=skm.confusion_matrix,
    ...    y_true=y_true,
    ...    y_pred=y_pred,
    ...    sensitive_features=group_membership_data
    ... )
    >>> mf_conf.overall
    array([[2, 4],
           [6, 6]], dtype=int64)
    >>> mf_conf.by_group
    sensitive_feature_0
    a    [[0, 2], [1, 1]]
    b    [[1, 0], [2, 3]]
    c    [[1, 2], [3, 2]]
    Name: confusion_matrix, dtype: object

Obviously for such cases, operations such as :meth:`MetricFrame.difference` have no meaning.
However, if scalar-returning metrics are also present, they will still be calculated:

.. doctest:: assessment_metrics
    :options:  +NORMALIZE_WHITESPACE

    >>> mf_conf_recall = MetricFrame(
    ...    metrics={ 'conf_mat':skm.confusion_matrix, 'recall':skm.recall_score },
    ...    y_true=y_true,
    ...    y_pred=y_pred,
    ...    sensitive_features=group_membership_data
    ... )
    >>> mf_conf_recall.overall
    conf_mat    [[2, 4], [6, 6]]
    recall                   0.5
    dtype: object
    >>> mf_conf_recall.by_group
                                 conf_mat  recall
    sensitive_feature_0
    a                    [[0, 2], [1, 1]]     0.5
    b                    [[1, 0], [2, 3]]     0.6
    c                    [[1, 2], [3, 2]]     0.4
    >>> mf_conf_recall.difference()
    conf_mat    None
    recall       0.2
    dtype: object

We see that the difference between group recall scores has been calculated, while a value of
:code:`None` has been returned for the meaningless 'maximum difference between two confusion matrices'
entry.

Inputs are Arrays of Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. _plot:

Plotting
--------

Plotting grouped metrics
^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way to visualize grouped metrics from the :class:`MetricFrame` is
to take advantage of the inherent plotting capabilities of
:class:`pandas.DataFrame`:

.. literalinclude:: ../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Analyze metrics using MetricFrame
    :end-before: # Customize plots with ylim

.. figure:: ../auto_examples/images/sphx_glr_plot_quickstart_001.png
    :target: auto_examples/plot_quickstart.html
    :align: center

It is possible to customize the plots. Here are some common examples.

Customize Plots: :code:`ylim`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The y-axis range is automatically set, which can be misleading, therefore it is
sometimes useful to set the `ylim` argument to define the yaxis range.

.. literalinclude:: ../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Customize plots with ylim
    :end-before: # Customize plots with colormap

.. figure:: ../auto_examples/images/sphx_glr_plot_quickstart_002.png
    :align: center


Customize Plots: :code:`colormap`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To change the color scheme, we can use the `colormap` argument. A list of colorschemes
can be found `here <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.

.. literalinclude:: ../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Customize plots with colormap
    :end-before: # Customize plots with kind

.. figure:: ../auto_examples/images/sphx_glr_plot_quickstart_003.png
    :align: center

Customize Plots: :code:`kind`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are different types of charts (e.g. pie, bar, line) which can be defined by the `kind`
argument. Here is an example of a pie chart.

.. literalinclude:: ../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Customize plots with kind
    :end-before: # Saving plots

.. figure:: ../auto_examples/images/sphx_glr_plot_quickstart_004.png
    :align: center

There are many other customizations that can be done. More information can be found in
:meth:`pandas.DataFrame.plot`.

In order to save a plot, access the :class:`matplotlib.figure.Figure` as below and save it with your
desired filename.

.. literalinclude:: ../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Saving plots

.. _dashboard:

Fairlearn dashboard
-------------------

The Fairlearn dashboard was a Jupyter notebook widget for assessing how a
model's predictions impact different groups (e.g., different ethnicities), and
also for comparing multiple models along different fairness and performance
metrics.

.. note::

    The :code:`FairlearnDashboard` is no longer being developed as
    part of Fairlearn.
    For more information on how to use it refer to
    `https://github.com/microsoft/responsible-ai-widgets <https://github.com/microsoft/responsible-ai-widgets>`_.
    Fairlearn provides some of the existing functionality through
    :code:`matplotlib`-based visualizations. Refer to the :ref:`plot` section.


References
----------

.. footbibliography::