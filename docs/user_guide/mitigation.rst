.. _mitigation:

Mitigation
==========

Fairlearn contains the following algorithms for mitigating unfairness:

.. list-table::
   :header-rows: 1
   :widths: 5 20 5 5 8
   :stub-columns: 1

   *  - algorithm
      - description
      - binary classification
      - regression
      - supported fairness definitions
   *  - :code:`fairlearn.` :code:`reductions.` :code:`ExponentiatedGradient`
      - A wrapper (reduction) approach to fair classification described in *A Reductions*
        *Approach to Fair Classification* [#2]_.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :code:`fairlearn.` :code:`reductions.` :code:`GridSearch`
      - A wrapper (reduction) approach described in Section 3.4 of *A Reductions*
        *Approach to Fair Classification* [#2]_. For regression it acts as a
        grid-search variant of the algorithm described in Section 5 of
        *Fair Regression: Quantitative Definitions and Reduction-based*
        *Algorithms* [#1]_.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :code:`fairlearn.` :code:`postprocessing.` :code:`ThresholdOptimizer`
      - Postprocessing algorithm based on the paper *Equality of Opportunity*
        *in Supervised Learning* [#3]_. This technique takes as input an
        existing classifier and the sensitive feature, and derives a monotone
        transformation of the classifier's prediction to enforce the specified
        parity constraints.
      - ✔
      - ✘
      - DP, EO, TPRP, FPRP

DP refers to *demographic parity*, EO to *equalized odds*, TPRP to *true positive
rate parity*, FPRP to *false positive rate parity*, ERP to *error rate parity*, and
BGL to *bounded group loss*. For
more information on the definitions refer to
:ref:`fairness_in_machine_learning`. To request additional algorithms or
fairness definitions, please open a
`new issue <https://github.com/fairlearn/fairlearn/issues>`_ on GitHub.

.. note::

   Fairlearn mitigation algorithms largely follow the
   `conventions of scikit-learn <https://scikit-learn.org/stable/developers/contributing.html#different-objects>`_,
   meaning that they implement the :code:`fit` method to train a model and the :code:`predict` method
   to make predictions. However, in contrast with 
   `scikit-learn <https://scikit-learn.org/stable/glossary.html#term-estimator>`_,
   Fairlearn algorithms can produce randomized predictors. Randomization of
   predictions is required to satisfy many definitions of fairness. Because of
   randomization, it is possible to get different outputs from the predictor's
   :code:`predict` method on identical data. For each of our algorithms, we provide
   explicit access to the probability distribution used for randomization.

.. _postprocessing:

Postprocessing
--------------

.. currentmodule:: fairlearn.postprocessing

.. _reductions:

Reductions
----------

.. currentmodule:: fairlearn.reductions

On a high level, the reduction algorithms within Fairlearn
enable unfairness mitigation for an arbitrary machine learning model with
respect to user-provided fairness constraints. All of the constraints currently supported
by reduction algorithms are group-fairness constraints. For more information on the
supported fairness constraints refer to :ref:`constraints_binary_classification`
and :ref:`constraints_regression`.

.. note::

   The choice of a fairness metric and fairness constraints is a crucial
   step in the AI development and deployment, and
   choosing an unsuitable constraint can lead to more harms.
   For a broader discussion of fairness as a
   sociotechnical challenge and how to view Fairlearn in this context refer to
   :ref:`fairness_in_machine_learning`.

The reductions approach for classification seeks to reduce binary
classification subject to fairness constraints to a sequence of weighted
classification problems (see [#2]_), and similarly for regression (see [#1]_).
As a result, the reduction algorithms
in Fairlearn only require a wrapper access to any "base" learning algorithm.
By this we mean that the "base" algorithm only needs to implement :code:`fit` and
:code:`predict` methods, as any standard scikit-learn estimator, but it
does not need to have any knowledge of the desired fairness constraints or sensitive features.

From an API perspective this looks as follows in all situations

>>> reduction = Reduction(base_estimator, constraints, **kwargs)  # doctest: +SKIP
>>> reduction.fit(X_train, y_train, sensitive_features=sensitive_features)  # doctest: +SKIP
>>> reduction.predict(X_test)  # doctest: +SKIP

Fairlearn doesn't impose restrictions on the referenced :code:`base_estimator`
other than the existence of :code:`fit` and :code:`predict` methods.
At the moment, the :code:`base_estimator`'s :code:`fit` method also needs to
provide a :code:`sample_weight` argument which the reductions techniques use
to reweight samples.
In the future Fairlearn will provide functionality to handle this even
without a :code:`sample_weight` argument.

Before looking more into reduction algorithms, this section
reviews the supported fairness constraints. All of them
are expressed as objects inheriting from the base class :code:`Moment`.
:code:`Moment`'s main purpose is to calculate the constraint violation of a
current set of predictions through its :code:`gamma` function as well as to
provide :code:`signed_weights` that are used to relabel and reweight samples.

.. _constraints_binary_classification:

Fairness constraints for binary classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All supported fairness constraints for binary classification inherit from
:code:`UtilityParity`. They are based on some underlying metric called
*utility*, which can be evaluated on individual data points and is averaged
over various groups of data points to form the *utility parity* constraint
of the form

.. math::

    \text{utility}_{a,e} = \text{utility}_e \quad \forall a, e

where :math:`a` is a sensitive feature value and :math:`e` is an *event*
identifier. Each data point has only one value of a sensitive feature,
and belongs to at most one event. In many examples, there is only
a single event :math:`*`, which includes all the data points. Other
examples of events include :math:`Y=0` and :math:`Y=1`. The utility
parity requires that the mean utility within each event equals
the mean utility of each group whose sensitive feature is :math:`a`
within that event.

The class :code:`UtilityParity` implements constraints that allow
some amount of violation of the utility parity constraints, where
the maximum allowed violation is specified either as a difference
or a ratio.

The *difference-based relaxation* starts out by representing
the utility parity constraints as pairs of
inequalities

.. math::

    \text{utility}_{a,e} - \text{utility}_{e} \leq 0 \quad \forall a, e\\
    -\text{utility}_{a,e} + \text{utility}_{e} \leq 0 \quad \forall a, e

and then replaces zero on the right-hand side
with a value specified as :code:`difference_bound`. The resulting
constraints are instantiated as

    >>> UtilityParity(difference_bound=0.01)  # doctest: +SKIP

Note that satisfying these constraints does not mean
that the difference between the groups with the highest and
smallest utility in each event is bounded by :code:`difference_bound`.
The value of :code:`difference_bound` instead bounds
the difference between the utility of each group and the overall mean
utility within each event. This, however,
implies that the difference between groups in each event is
at most twice the value of :code:`difference_bound`.

The *ratio-based relaxation* relaxes the parity
constraint as

.. math::

    r \leq \dfrac{\text{utility}_{a,e}}{\text{utility}_e} \leq \dfrac{1}{r} \quad \forall a, e

for some value of :math:`r` in (0,1]. For example, if :math:`r=0.9`, this means
that within each event
:math:`0.9 \cdot \text{utility}_{a,e} \leq \text{utility}_e`, i.e., the utility for
each group needs to be at least 90% of the overall utility for the event, and
:math:`0.9 \cdot \text{utility}_e \leq \text{utility}_{a,e}`, i.e., the overall utility
for the event needs to be at least 90% of each group's utility.

The two ratio constraints can be rewritten as

.. math::

   - \text{utility}_{a,e} + r \cdot \text{utility}_e \leq 0 \quad \forall a, e \\
   r \cdot \text{utility}_{a,e} - \text{utility}_e \leq 0 \quad \forall a, e

When instantiating the ratio constraints, we use :code:`ratio_bound` for :math:`r`,
and also allow further relaxation by replacing the zeros on the right hand side
by some non-negative :code:`ratio_bound_slack`. The resulting instantiation
looks as

    >>> UtilityParity(ratio_bound=0.9, ratio_bound_slack=0.01)  # doctest: +SKIP

Similarly to the difference constraints, the ratio constraints do not directly
bound the ratio between the pairs of groups, but such a bound is implied.

.. note::

    It is not possible to specify both :code:`difference_bound` *and*
    :code:`ratio_bound` for the same constraint object.

.. _demographic_parity:

Demographic Parity
^^^^^^^^^^^^^^^^^^

A binary classifier :math:`h(X)` satisfies *demographic parity* if

.. math::
    
    \P[h(X) = 1 \given A = a] = \P[h(X) = 1] \quad \forall a
 
In other words, the selection rate or percentage of samples with label 1
should be equal across all groups. Implicitly this means the percentage
with label 0 is equal as well. In this case, the utility function
is equal to :math:`h(X)` and there is only a single event :math:`*`.

In the example below group :code:`"a"` has a selection rate of 60%,
:code:`"b"` has a selection rate of 20%. The overall selection rate is 40%,
so :code:`"a"` is `0.2` above the overall selection rate, and :code:`"b"` is
`0.2` below. Invoking the method :code:`gamma` shows the values
of the left-hand sides of the constraints described
in :ref:`constraints_binary_classification`, which is independent
of the provided :code:`difference_bound`. Note that the left-hand sides
corresponding to different values of :code:`sign` are just negatives
of each other.
The value of :code:`y_true` is in this example irrelevant to the calculations,
because the underlying utility in demographic parity, selection rate, does not
consider performance relative to the true labels, but rather proportions in
the predicted labels.

.. note::

    When providing :code:`DemographicParity` to mitigation algorithms, only use
    the constructor and the mitigation algorithm itself then invokes :code:`load_data`.
    The example below uses :code:`load_data` to illustrate how :code:`DemographicParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

.. doctest:: mitigation

    >>> from fairlearn.reductions import DemographicParity
    >>> from fairlearn.metrics import selection_rate_group_summary
    >>> import numpy as np
    >>> dp = DemographicParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> selection_rate_group_summary(y_true, y_pred, sensitive_features=sensitive_features)
    {'overall': 0.4, 'by_group': {'a': 0.6, 'b': 0.2}}
    >>> dp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.2
                 b          -0.2
    -     all    a          -0.2
                 b           0.2
    dtype: float64
 
The ratio constraints for the demographic parity with :code:`ratio_bound`
:math:`r` (and :code:`ratio_bound_slack=0`) take form

.. math::

    r \leq \dfrac{\P[h(X) = 1 \given A = a]}{\P[h(X) = 1]} \leq \dfrac{1}{r} \quad \forall a

Revisiting the same example as above we get

.. doctest:: mitigation

    >>> dp = DemographicParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> dp.load_data(X, y_pred, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.14
                 b          -0.22
    -     all    a          -0.24
                 b           0.16
    dtype: float64

Following the expressions for the left-hand sides
of the constraints, we obtain

.. math::

    r \cdot \text{utility}_{a,*} - \text{utility}_* = 0.9 \times 0.6 - 0.4 = 0.14 \\
    r \cdot \text{utility}_{b,*} - \text{utility}_* = 0.9 \times 0.2 - 0.4 = -0.22 \\
    - \text{utility}_{a,*} + r \cdot \text{utility}_* = - 0.6 + 0.9 \times 0.4 = -0.24 \\
    - \text{utility}_{b,*} + r \cdot \text{utility}_* = - 0.2 + 0.9 \times 0.4 = 0.16 \\

.. _true_positive_rate_parity:
.. _false_positive_rate_parity:

True Positive Rate Parity and False Positive Rate Parity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A binary classifier :math:`h(X)` satisfies *true positive rate parity* if

.. math::

    \P[h(X) = 1 \given A = a, Y = 1] = \P[h(X) = 1 \given Y = 1] \quad \forall a

and *false positive rate parity* if

.. math::

    \P[h(X) = 1 \given A = a, Y = 0] = \P[h(X) = 1 \given Y = 0] \quad \forall a

In first case, we only have one event :math:`Y=1` and
ignore the samples with :math:`Y=0`, and in the second case vice versa.
Refer to :ref:`equalized_odds` for the fairness constraint type that simultaneously
enforce both true positive rate parity and false positive rate parity
by considering both events :math:`Y=0` and :math:`Y=1`.

In practice this can be used in a difference-based relaxation as follows:

.. doctest:: mitigation

    >>> from fairlearn.reductions import TruePositiveRateParity
    >>> from fairlearn.metrics import true_positive_rate_group_summary
    >>> import numpy as np
    >>> tprp = TruePositiveRateParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  1,   1 ,  1 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  1 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> true_positive_rate_group_summary(y_true, y_pred, sensitive_features=sensitive_features)
    {'overall': 0.571..., 'by_group': {'a': 0.75, 'b': 0.333...}}
    >>> tprp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> tprp.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=1  a           0.178571
                   b          -0.238095
    -     label=1  a          -0.178571
                   b           0.238095
    dtype: float64

.. note::

    When providing :code:`TruePositiveRateParity` or :code:`FalsePositiveRateParity`
    to mitigation algorithms, only use
    the constructor. The mitigation algorithm itself then invokes :code:`load_data`.
    The example uses :code:`load_data` to illustrate how :code:`TruePositiveRateParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

Alternatively, a ratio-based relaxation is also available:

.. doctest:: mitigation

    >>> tprp = TruePositiveRateParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> tprp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> tprp.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=1  a           0.103571
                   b          -0.271429
    -     label=1  a          -0.235714
                   b           0.180952
    dtype: float64

.. _equalized_odds:
    
Equalized Odds
^^^^^^^^^^^^^^

A binary classifier :math:`h(X)` satisfies *equalized odds* if it satisfies both
*true positive rate parity* and *false positive rate parity*, i.e.,

.. math::

    \P[h(X) = 1 \given A = a, Y = y] = \P[h(X) = 1 \given Y = y] \quad \forall a, y

The constraints represent the union of constraints for true positive rate
and false positive rate.

.. doctest:: mitigation

    >>> from fairlearn.reductions import EqualizedOdds
    >>> eo = EqualizedOdds(difference_bound=0.01)
    >>> eo.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> eo.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=0  a          -0.333333
                   b           0.166667
          label=1  a           0.178571
                   b          -0.238095
    -     label=0  a           0.333333
                   b          -0.166667
          label=1  a          -0.178571
                   b           0.238095
    dtype: float64

.. _error_rate_parity:

Error Rate Parity
^^^^^^^^^^^^^^^^^

The *error rate parity* requires that the error rates should be
the same across all groups. For a classifier :math:`h(X)`
this means that

.. math::

   \P[h(X) \ne Y \given A = a] = \P[h(X) \ne Y] \quad \forall a

In this case, the utility is equal to 1 if :math:`h(X)\ne Y` and equal to
0 if :math:`h(X)=Y`, and so large value of utility here actually correspond
to poor outcomes. The difference-based relaxation specifies that
the error rate of any given group should not deviate from
the overall error rate by more than the value of :code:`difference_bound`.

.. doctest:: mitigation

    >>> from fairlearn.reductions import ErrorRateParity
    >>> from fairlearn.metrics import accuracy_score_group_summary
    >>> accuracy_score_group_summary(y_true, y_pred, sensitive_features=sensitive_features)
    {'overall': 0.6, 'by_group': {'a': 0.8, 'b': 0.4}}
    >>> erp = ErrorRateParity(difference_bound=0.01)
    >>> erp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> erp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a          -0.2
                 b           0.2
    -     all    a           0.2
                 b          -0.2
    dtype: float64

.. note::

    When providing :code:`ErrorRateParity` to mitigation algorithms, only use
    the constructor. The mitigation algorithm itself then invokes :code:`load_data`.
    The example uses :code:`load_data` to illustrate how :code:`ErrorRateParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

Alternatively, error rate parity can be relaxed via ratio constraints as

.. math::

   r \leq \dfrac{\P[h(X) \ne Y \given A = a]}{\P[h(X) \ne Y]} \leq \dfrac{1}{r} \quad \forall a

with a :code:`ratio_bound` :math:`r`. The usage is identical with other
constraints:

.. doctest:: mitigation

    >>> from fairlearn.reductions import ErrorRateParity
    >>> erp = ErrorRateParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> erp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> erp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a          -0.22
                 b           0.14
    -     all    a           0.16
                 b          -0.24
    dtype: float64

.. _constraints_multi_class_classification:

Fairness constraints for multi-class classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reductions approaches do not support multi-class classification yet at this
point. If this is an important scenario for you please let us know!

.. _constraints_regression:

Fairness constraints for regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance objective in the regression scenario is to minimize the
loss of our regressor :math:`h`. The loss can be expressed as
:class:`SquareLoss` or :class:`AbsoluteLoss`. Both take constructor arguments
:code:`min_val` and :code:`max_val` that define the value range within which
the loss is evaluated. Values outside of the value range get clipped.

.. doctest:: mitigation

    >>> from fairlearn.reductions import SquareLoss, AbsoluteLoss, ZeroOneLoss
    >>> y_true = [0,   0.3, 1,   0.9]
    >>> y_pred = [0.1, 0.2, 0.9, 1.3]
    >>> SquareLoss(0, 2).eval(y_true, y_pred)
    array([0.01, 0.01, 0.01, 0.16])
    >>> # clipping at 1 reduces the error for the fourth entry
    >>> SquareLoss(0, 1).eval(y_true, y_pred)
    array([0.01, 0.01, 0.01, 0.01])
    >>> AbsoluteLoss(0, 2).eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.4])
    >>> AbsoluteLoss(0, 1).eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.1])
    >>> # ZeroOneLoss is identical to AbsoluteLoss(0, 1)
    >>> ZeroOneLoss().eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.1])

When using Fairlearn's reduction techniques for regression it's required to
specify the type of loss by passing the corresponding loss object when
instantiating the object that represents our fairness constraint. The only
supported type of constraint at this point is :class:`BoundedGroupLoss`.

.. _bounded_group_loss:

Bounded Group Loss
^^^^^^^^^^^^^^^^^^

*Bounded group loss* requires the loss of each group to be below a
user-specified amount :math:`\zeta`. If :math:`\zeta` is chosen reasonably
small the losses of all groups are very similar.
Formally, a predictor :math:`h` satisfies bounded group loss at level
:math:`\zeta` under a distribution over :math:`(X, A, Y)` if

.. math::

    \E[loss(Y, h(X)) \given A=a] \leq \zeta \quad \forall a

In the example below we use :class:`BoundedGroupLoss` with
:class:`ZeroOneLoss` on two groups :code:`"a"` and :code:`"b"`.
Group :code:`"a"` has an average loss of :math:`0.05`, while group
:code:`"b"`'s average loss is :math:`0.5`.

.. doctest:: mitigation

    >>> from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss
    >>> from fairlearn.metrics import mean_absolute_error_group_summary
    >>> bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
    >>> X                  = np.array([[0], [1], [2], [3]])
    >>> y_true             = np.array([0.3, 0.5, 0.1, 1.0])
    >>> y_pred             = np.array([0.3, 0.6, 0.6, 0.5])
    >>> sensitive_features = np.array(["a", "a", "b", "b"])
    >>> mean_absolute_error_group_summary(y_true, y_pred, sensitive_features=sensitive_features)
    {'overall': 0.275, 'by_group': {'a': 0.0499..., 'b': 0.5}}
    >>> bgl.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> bgl.gamma(lambda X: y_pred)
    group_id
    a    0.05
    b    0.50
    Name: loss, dtype: float64

.. note::

    In the example above the :code:`BoundedGroupLoss` object does not use the
    :code:`upper_bound` argument. It is only used by reductions techniques
    during the unfairness mitigation. As a result the constraint violation
    detected by :code:`gamma` is identical to the mean absolute error.

.. _exponentiated_gradient:

Exponentiated Gradient
~~~~~~~~~~~~~~~~~~~~~~

.. _grid_search:

Grid Search
~~~~~~~~~~~

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
      NeurIPS, 2016.
