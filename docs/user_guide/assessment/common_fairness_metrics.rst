.. _common_fairness_metrics:

Common fairness metrics
=======================

.. currentmodule:: fairlearn.metrics

In the sections below, we review the most common fairness metrics, as well
as their underlying assumptions and suggestions for use. Each metric requires
that some aspects of the predictor behavior be comparable across groups.

.. note::
  Note that *common* usage does not imply *correct* usage; we discuss
  one very common misuse in the
  :ref:`section on the Four-Fifths Rule <assessment_four_fifths>`

In the code examples presented below, we will use the following input arrays:

.. doctest:: common_fairness_metrics_code

    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> sf_data = ['b', 'b', 'a', 'b', 'b', 'c', 'c', 'c', 'a',
    ...            'a', 'c', 'a', 'b', 'c', 'c', 'b', 'c', 'c']



.. _assessment_demographic_parity:

Demographic parity
------------------

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
   Demographic parity is also sometimes referred to as *independence*,
   *group fairness*, *statistical parity*, and *disparate impact*.

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
See :ref:`assessment_intersecting_groups` below to see how :class:`MetricFrame` can
help with this.

However, more granular categories generally contain smaller sample sizes, and 
it can be more difficult to establish that trends seen in very small 
samples are not due to random chance. 
We also recommend watching out for the
`multiple comparisons problem <https://stats.libretexts.org/Bookshelves/Applied_Statistics/Book%3A_Biological_Statistics_(McDonald)/06%3A_Multiple_Tests/6.01%3A_Multiple_Comparisons>`_, 
which states that the more statistical inferences are made, the 
more erroneous those inferences will become. For example, in the case 
of evaluating fairness metrics on multiple groups, as we break the 
groups down into more granular categories and evaluate those smaller 
groups, it will become more likely that these subgroups will 
differ enough to fail one of the metrics. For dealing with the multiple 
comparisons problem, we recommend investigating
`statistical techniques <https://www.statology.org/bonferroni-correction/>`_ 
meant to correct the errors produced by individual statistical tests. 

Fairlearn provides the :func:`demographic_parity_difference` and
:func:`demographic_parity_ratio` functions for computing demographic
parity measures for binary classification data, both of which return
a scalar result.
The first reports the absolute difference between the highest and
lowest selection rates :math:`a \in A` so a result of 0 indicates
that demographic parity has been achieved.
The second reports the ratio of the lowest and highest selection rates,
so a result of 1 means there is demographic parity.
This metric can potentially be used to implement the 'Four-Fifths' Rule,
but :ref:`read our discussion below <assessment_four_fifths>` to understand whether this is an appropriate metric for your use case.
As with any fairness metric, achieving demographic parity does *not* automatically mean
that the classifier is fair!


.. doctest:: common_fairness_metrics_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import demographic_parity_difference
    >>> print(demographic_parity_difference(y_true,
    ...                                     y_pred,
    ...                                     sensitive_features=sf_data))
    0.25
    >>> from fairlearn.metrics import demographic_parity_ratio
    >>> print(demographic_parity_ratio(y_true,
    ...                                y_pred,
    ...                                sensitive_features=sf_data))
    0.66666...


.. _assessment_equalized_odds:

Equalized odds
--------------

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


.. doctest:: common_fairness_metrics_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import equalized_odds_difference
    >>> print(equalized_odds_difference(y_true,
    ...                                 y_pred,
    ...                                 sensitive_features=sf_data))
    1.0
    >>> from fairlearn.metrics import equalized_odds_ratio
    >>> print(equalized_odds_ratio(y_true,
    ...                            y_pred,
    ...                            sensitive_features=sf_data))
    0.0



.. _assessment_equal_opportunity:

Equal opportunity
-----------------

Equal opportunity is a relaxed version of equalized odds that only considers
conditional expectations with respect to positive labels, i.e., :math:`Y=1`.
:footcite:p:`hardt2016equality`
Another way of thinking about this metric is 
requiring equal outcomes only within the subset of records belonging to the 
positive class. In the hiring example, equal opportunity requires that the 
individuals in *group A* who are qualified to be hired are just as likely to 
be chosen as individuals in *group B* who are qualified to be hired. 
However, by not considering whether false 
positive rates are equivalent across groups, equal opportunity does not 
capture the costs of missclassification disparities.



.. _assessment_four_fifths:

The Four Fifths Rule: Often Misapplied
--------------------------------------

In the literature around fairness in machine learning, one will often find
the so-called "four fifths rule" or "80% rule" used to assess whether a model
(or mitigation technique) has produced a 'fair' result.
Typically, the rule is implemented by using the demographic parity ratio introduced
in the :ref:`assessment_demographic_parity` section above
(within Fairlearn, one can use :func:`demographic_parity_ratio`), with a result
considered 'fair' if the ratio exceeds 80% for all identified subgroups.
*Application of this threshold is wrong in many scenarios.*

As we note in many other places in the Fairlearn documentation, 'fairness'
must be assessed by examining the entire sociotechnical context of a machine
learning system.
In particular, it is important to start from the harms which can occur to real
people, and work inwards towards the model.
The demographic parity ratio is simply a metric by which a particular model
may be measured (on a particular dataset).
Given the origin of the 'four-fifths rule' (which we will discuss next), its
application may also give an unjustified feeling of legal invulnerability by
conflating fairness with legality.
In reality, 'fairness' is not always identical to 'legally allowable,' and
the former may not even be a strict subset of the latter. [#f1]_

The 'four fifths rule' has its origins in a specific area of US
federal employment law.
It is a limit for
`prima facie evidence <https://en.wikipedia.org/wiki/Prima_facie>`_
that illegal discrimination has occurred relative to a 
relevant control population.
The four-fifths rule is one of many test statistics that can be used
to establish a *prima facie* case, but it is generally only used
within the context of
`US Federal employment regulation <https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XIV/part-1607>`_. 
A violation of the rule is still not sufficient to demonstrate that
illegal discrimination has occurred - a causal link between the
statistic and alleged discrimination must still be shown, and the
(US) court would examine the particulars of each case.
For an example of the subtleties involved, see
`Ricci v. Stefano <https://en.wikipedia.org/wiki/Ricci_v._DeStefano>`_
which resulted from an attempt to 'correct' for disparate impact.
*Outside* its particular context in US federal employment law,
the 'four fifths rule' has no validity and its misapplication
is an example of the :ref:`portability trap <portability_trap>`.

Taken together, we see that applying the 'four fifths rule' will
not be appropriate in most cases.
Even in cases where it is applicable, the rule does not automatically
avoid legal jeopardy, much less ensure that results are fair.
The use of the 'four fifths rule' in this manner is an indefensible
example of epistemic trespassing. [#f2]_
It is for this reason that we try to avoid the use of legal
terminology in our documentation.

For a much deeper discussion of the issues involved, we suggest
:footcite:ct:`watkins2022fourfifths`.
A higher level look at how legal concepts of fairness can collide
with mathematical measures of disparity, see
:footcite:ct:`Xiang2019legalcompatibility`.


Summary
-------

We have introduced three commonly used fairness metrics in this section,
which can be summed up as follows:

* Demographic Parity

    * *What it compares:* Predictions between different groups
      (true values are ignored)

    * *Reason to use:* If the input data are known to contain
      biases, demographic parity may be appropriate to measure fairness

    * *Caveats:* By only using the predicted values, information is thrown
      away. The selection rate is also a very coarse measure of the
      distribution between groups, making it tricky to use as an optimization
      constraint

* Equalized Odds

    * *What it compares:* True and False Positive rates between different groups

    * *Reason to use:* If historical data does not contain measurement bias or historical
      bias that we need to take into account, and true and false positives
      are considered to be (roughly) of the same importance, equalized odds may be useful

    * *Caveats:* If there are historical biases in the data, then the original labels
      may hold little value. A large imbalance between the positive and negative classes
      will also accentuate any statistical issues related to sensitive groups with low
      membership

* Equal opportunity

    * *What it compares:* True Positive rates between different groups

    * *Reason to use:* If historical data are useful, and extra false positives
      are much less likely to cause harm than missed true positives, equal
      opportunity may be useful

    * *Caveats:* If there are historical biases in the data, then the original labels
      may hold little value. A large imbalance between the positive and negative classes
      will also accentuate any statistical issues related to sensitive groups with low
      membership


However, the fact these are common metrics does not make them applicable to any given
situation.
In particular, :ref:`demographic parity is often misapplied <assessment_four_fifths>`.


References
----------

.. footbibliography::


.. rubric:: Footnotes

.. [#f1] For a related example, see the discussion on 'law' and 'justice' in
         *The Caves of Steel* (Asimov, 1953)

.. [#f2] Epistemic trespassing is the process of taking expertise in one field and
         applying it to another in which one does not have an equivalent (or any)
         competence.
         This is not an intrinsically bad thing - one could label all
         interdisciplinary research a form of epistemic trespassing.
         However, doing so successfully requires a willingness to learn the subtleties
         of the new field.
