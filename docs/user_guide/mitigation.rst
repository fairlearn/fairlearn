Mitigation
==========

Fairlearn contains the following algorithms for mitigating unfairness:

.. list-table:: Unfairness mitigation algorithms
   :header-rows: 1
   :widths: 5 20 5 5 8
   :stub-columns: 1

   *  - algorithm
      - description
      - binary classification
      - regression
      - supported fairness definitions
   *  - :code:`fairlearn.` :code:`reductions.` :code:`ExponentiatedGradient`
      - Black-box approach to fair classification described in *A Reductions*
        *Approach to Fair Classification* [#2]_.
      - ✔
      - ✘ :superscript:`*`
      - DP, EO, TPRP, ERR
   *  - :code:`fairlearn.` :code:`reductions.` :code:`GridSearch`
      - Black-box approach described in Section 3.4 of *A Reductions*
        *Approach to Fair Classification* [#2]_. For regression it acts as a
        grid-search variant of the algorithm described in Section 5 of
        *Fair Regression: Quantitative Definitions and Reduction-based*
        *Algorithms* [#1]_.
      - ✔
      - ✔
      - DP, EO, TPRP, ERR, BGL
   *  - :code:`fairlearn.` :code:`postprocessing.` :code:`ThresholdOptimizer`
      - Postprocessing algorithm based on the paper *Equality of Opportunity*
        *in Supervised Learning* [#3]_. This technique takes as input an
        existing classifier and the sensitive feature, and derives a monotone
        transformation of the classifier's prediction to enforce the specified
        parity constraints.
      - ✔
      - ✘
      - DP, EO

.. [*] coming soon!

DP refers to demographic parity, EO to equalized odds, TPRD to true positive
rate difference, ERR to error rate ratio, and BGL to bounded group loss. For
more information on the definitions refer to
:ref:`fairness_in_machine_learning`. To request additional algorithms or
fairness definitions, please open a
`new issue <https://github.com/fairlearn/fairlearn/issues>`_ on GitHub.

.. note: Randomized predictors

The Fairlearn package largely follows the
`terminology established by scikit-learn <https://scikit-learn.org/stable/developers/contributing.html#different-objects>`_,
specifically:

* *Estimators* implement a :code:`fit` method.
* *Predictors* implement a :code:`predict` method.

**Randomization.** In contrast with 
`scikit-learn <https://scikit-learn.org/stable/glossary.html#term-estimator>`_,
estimators in Fairlearn can produce randomized predictors. Randomization of
predictions is required to satisfy many definitions of fairness. Because of
randomization, it is possible to get different outputs from the predictor's
:code:`predict` method on identical data. For each of our methods, we provide
explicit access to the probability distribution used for randomization.

.. _postprocessing:

Postprocessing
--------------

.. currentmodule:: fairlearn.postprocessing

.. _reductions:

Reductions
----------

.. currentmodule:: fairlearn.reductions

From a very high-level perspective the reductions techniques within Fairlearn
enable unfairness mitigation on an arbitrary machine learning model with
respect to a user provided fairness objective. For more information on the
supported kinds of fairness refer to :ref:`constraints_binary_classification`
and :ref:`constraints_regression`. All of the constraints currently supported
by reductions methods are group fairness constraints.
Note that the choice of fairness constraint is crucial to the outcome, and
choosing an unsuitable constraint can cause or worsen harms on individuals or
groups affected by a decision. For a broader discussion of fairness as a
sociotechnical challenge and how to view Fairlearn in this context refer to
:ref:`fairness_in_machine_learning`.

The reductions approach is based on the idea in [#2]_ to reduce binary
classification subject to fairness constraints to a sequence of cost-sensitive
classification problems. As a result, the reductions methods in Fairlearn
require only black-box access to a cost-sensitive classification algorithm,
which does not need to have any knowledge of the desired definition of
fairness or sensitive features. [#1]_ extends this to regression scenarios.

From an API perspective this looks as follows in all situations

>>> reduction = Reduction(estimator, constraints, **kwargs)
>>> reduction.fit(X_train, y_train, sensitive_features=sensitive_features)
>>> reduction.predict(X_test)

Fairlearn doesn't impose restrictions on the referenced :code:`estimator`
other than the existence of :code:`fit` and :code:`predict` methods.
At the moment, :code:`fit` also needs to provide a :code:`sample_weight`
argument which the reductions techniques use to reweight samples.
In the future Fairlearn will provide functionality to handle this even
without a :code:`sample_weight` argument.

Before looking more into the reductions approaches themselves this section
examines the different constraints that are compatible with them. All of them
are expressed as moments inheriting from :code:`ClassificationMoment` or
:code:`LossMoment`, which in turn inherit from the base class :code:`Moment`.
:code:`Moment`'s main purpose is to calculate the constraint violation of a
current set of predictions through its :code:`gamma` function as well as to
provide :code:`signed_weights` that are used to relabel and reweight samples.

.. _constraints_binary_classification:

Constraints for binary classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All existing constraints for binary classification inherit from
:code:`ClassificationMoment` and can be formulated as difference or ratio-based
constraints.

Difference-based constraints consider the difference with respect
to a metric between a group of samples and the overall population or
cohorts thereof.
Parity represents the state where

.. math::

    \text{metric}_a - \text{metric}_* = 0 \; \forall a

where :math:`a` is a group indentifier and :math:`*` represents the overall
population. This is internally represented as two inequalities

.. math::

    \text{metric}_a - \text{metric}_* \leq 0 \; \forall a
    -\text{metric}_a + \text{metric}_* \leq 0 \; \forall a

The algorithms do not consider every kind of violation of these constraints,
but rather violations that go beyond their :code:`difference_bound`. Such
a difference-based constraint could be instantiated as

    >>> ClassificationMoment(difference_bound=0.01)

Note that achieving parity with the :code:`difference_bound` does not mean
that the difference between the group with the highest metric value and the
group with the lowest metric value is below the :code:`difference_bound`, but
rather the difference between each individual group and the metric value for
the overall population (or cohort thereof) is below :code:`difference_bound`. 

In constrast, ratio-based constraints consider the ratio as opposed to the
difference. Parity is represented as

.. math::

    r \leq \dfrac{\text{metric}_a}{\text{metric}_*} \leq \dfrac{1}{r} \; \forall a

with a :code:`ratio_bound` :math:`r`. Similar to the :code:`difference_bound`
for difference-based constraints, ratio-based constraints have a
:code:`ratio_bound_slack`. The two inequalities can also be considered
separately as

.. math::

   r \leq \dfrac{\text{metric}_a}{\text{metric}_*} \; \forall a
   \dfrac{\text{metric}_a}{\text{metric}_*} \leq \dfrac{1}{r} \; \forall a

and one step further in the equivalent form:

.. math::

   - \text{metric}_a + r \text{metric}_* \leq 0 \; \forall a
   r \text{metric}_a - \text{metric}_* \leq 0 \; \forall a

To use a constraint in its ratio-based version simply indicate as much through
its arguments

    >>>> ClassificationMoment(ratio_bound=0.9, ratio_bound_slack=0.01)

In other words, this means that
:math:`0.9 \text{metric}_a \leq \text{metric}_*`, i.e., the metric value for
each group needs to be at least 90% of the overall metric value, and
:math:`0.9 \text{metric}_* \leq \text{metric}_a`, i.e., the overall metric
value needs to be at least 90% of each group's metric value.

It is not possible to specify both :code:`difference_bound` *and*
:code:`ratio_bound` for the same constraint object.

.. _demographic_parity:

Demographic Parity
^^^^^^^^^^^^^^^^^^

A classifier :math:`h(X)` satisfies *Demographic Parity* if

.. math::
    
    P[h(X) = y' \given A = a] = P[h(X) = y'] \; \forall a, y'
 
In other words, the selection rate or percentage of samples with label 1
should be equal between all groups. Implicitly this means the percentage
with label 0 is equal as well.

In the example below class :code:`"a"` has a selection rate of 60%,
:code:`"b"` has a selection rate of 20%. The overall selection rate is 40%,
so :code:`"a"` is `0.2` above the overall selection rate, and :code:`"b"` is
`0.2` below. The seemingly redundantly stored constraint violation with both
:code:`"a"` and :code:`"b"` showing up twice stems from the constraint
definition outlined in :ref:`constraints_binary_classification`. Each equality
constraint is written as two inequalities of which one contains
:math:`\text{metric}_a` positively (sign :math:`+`) and one negatively
(sign :math:`-`).
Note that the specified :code:`difference_bound` is not used
since the constraint object is not used in a mitigation in this example.
Similarly, :code:`y_true` is technically irrelevant to the calculations
because the underlying metric of Demographic Parity, selection rate, does not
consider performance relative to the true labels, but rather proportions in
the predicted labels.
 
    >>> from fairlearn.reductions import DemographicParity
    >>> from fairlearn.metrics import selection_rate_group_summary
    >>> import numpy as np
    >>> dp = DemographicParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> selection_rate_group_summary(y_pred, y_pred, sensitive_features=sensitive_features)
    {'overall': 0.4, 'by_group': {'a': 0.6, 'b': 0.2}}
    >>> dp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.2
                 b          -0.2
    -     all    a          -0.2
                 b           0.2
    dtype: float64
 
Rewriting this into a ratio-based constraint with :code:`ratio_bound`
:math:`r` we get

.. math::

    r \leq \dfrac{P[h(X) = y' \given A = a]}{P[h(X) = y']} \leq \dfrac{1}{r} \; \forall a, y'

Revisiting the same example from above we get

    >>> dp = DemographicParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> dp.load_data(X, y_pred, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.14
                 b          -0.22
    -     all    a          -0.24
                 b           0.16
    dtype: float64

Just like :code:`difference_bound` before :code:`ratio_bound_slack` does not
affect the outcome here, because no mitigation is applied in the example.
This time the constraint violation is calculated using the :code:`ratio_bound`
`0.9` and again with the definition from
:ref:`constraints_binary_classification` and selection rate as the used
metric:

.. math::

    r \text{metric}_a - \text{metric}_* = 0.9 \times 0.6 - 0.4 = 0.14
    r \text{metric}_b - \text{metric}_* = 0.9 \times 0.2 - 0.4 = -0.22
    - \text{metric}_a + r \text{metric}_* = - 0.6 + 0.9 \times 0.4 = -0.24
    - \text{metric}_b + r \text{metric}_* = - 0.2 + 0.9 \times 0.4 = 0.16

.. _true_positive_rate_parity:

True Positive Rate Parity
^^^^^^^^^^^^^^^^^^^^^^^^^

A classifier :math:`h(X)` satisfies *True Positive Rate Parity* if

.. math::

    P[h(X) = 1 \given A = a, Y = 1] = P[h(X) = 1 \given Y = 1] \; \forall a

Note that this formulation ignores what happens to samples with :math:`Y=0`.
Refer to :ref:`equalized_odds` for a constraint that extends this by
considering all samples, i.e., the samples with :math:`Y=0` as well as the
ones with :math:`Y=1`.

In practice this can be used in a difference-based version as follows:
    
    >>> from fairlearn.reductions import TruePositiveRateParity
    >>> from fairlearn.metrics import true_positive_rate_group_summary
    >>> import numpy as np
    >>> tprp = TruePositiveRateParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  1,   1 ,  1 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> tprp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> true_positive_rate_group_summary(y_true, y_pred, sensitive_features=sensitive_features)
    {'overall': 0.5714285714285714, 'by_group': {'a': 0.75, 'b': 0.3333333333333333}}
    >>> tprp.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=1  a           0.178571
                   b          -0.238095
    -     label=1  a          -0.178571
                   b           0.238095
    dtype: float64

    
.. _equalized_odds:
    
Equalized Odds
^^^^^^^^^^^^^^

A classifier :math:`h(X)` satisfies *Equalized Odds* if it satisfies both
*True Positive Rate Parity* and *False Positive Rate Parity*, i.e.,

.. math::

    P[h(X) = y' \given A = a, Y = y] = P[h(X) = y' \given Y = y] \; \forall a, y, y'

.. _error_rate_parity:

Error Rate Parity
^^^^^^^^^^^^^^^^^

The basic idea behind *Error Rate Parity* is that the error rate should be
similar between different groups of people.

A classifier :math:`h(X)` satisfies *Error Rate Parity* if it satisfies

.. math::

   \E[\lvert h(X) - Y \rvert \given A = a] - \E[\lvert h(X) - Y \rvert] \leq 0 \; \forall a

This specifies that the error rate of any given group should not deviate from
the overall error rate by more a :code:`difference_bound`.
Similarly, *Error Rate Parity* can be expressed through a ratio constraint as

.. math::

   r <= \dfrac{\E[\lvert h(X) - Y \rvert \given A = a]}{\E[\lvert h(X) - Y \rvert]} <= \dfrac{1}{r} \; \forall a

with a :code:`ratio_bound` :math:`r`. A similar kind of slack to the
:code:`difference_bound` from the difference-based formulation is applied in
this ratio-based version, but it is referred to as :code:`ratio_bound_slack`.

.. _constraints_multi_class_classification:

Constraints for multi-class classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reductions approaches do not support multi-class classification yet at this
point. If this is an important scenario for you please let us know!

.. _constraints_regression:

Constraints for regression
~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance objective in the regression scenario is to minimize the
loss of our classifier :math:`h`.

BoundedGroupLoss
^^^^^^^^^^^^^^^^



Exponentiated Gradient
^^^^^^^^^^^^^^^^^^^^^^

Grid Search
^^^^^^^^^^^

.. topic:: References:

   .. [#1] Agarwal, Dudik, Wu `"Fair Regression: Quantitative Definitions and
      Reduction-based Algorithms" <https://arxiv.org/pdf/1905.12843.pdf>`_,
      ICML, 2019.
   
   .. [#2] Agarwal, Beygelzimer, Dudik, Langford, Wallach `"A Reductions
      Approach to Fair Classification"
      <https://arxiv.org/pdf/1803.02453.pdf>`_, ICML, 2018.
   
   .. [#3] Hardt, Price, Srebro `"Equality of Opportunity in Supervised
      Learning"
      <https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf>`_,
      NIPS, 2016.
