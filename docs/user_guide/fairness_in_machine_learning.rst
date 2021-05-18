.. _fairness_in_machine_learning:
.. _terminology:

Fairness in Machine Learning
============================

Fairness of AI systems
----------------------

AI systems can behave unfairly for a variety of reasons. Sometimes it is
because of societal biases reflected in the training data and in the decisions
made during the development and deployment of these systems. In other cases,
AI systems behave unfairly not because of societal biases, but because of
characteristics of the data (e.g., too few data points about some group of
people) or characteristics of the systems themselves. It can be hard to
distinguish between these reasons, especially since they are not mutually
exclusive and often exacerbate one another. Therefore, we define whether an AI
system is behaving unfairly in terms of its impact on people — i.e., in terms
of harms — and not in terms of specific causes, such as societal biases, or in
terms of intent, such as prejudice.

**Usage of the word bias.** Since we define fairness in terms of harms
rather than specific causes (such as societal biases), we avoid the usage of
the words *bias* or *debiasing* in describing the functionality of Fairlearn.

Types of harms
--------------

There are many types of harms (see, e.g., the
`keynote by K. Crawford at NeurIPS 2017 <https://www.youtube.com/watch?v=fMym_BKWQzk>`_).
The Fairlearn package is most applicable to two kinds of harms:

* *Allocation harms* can occur when AI systems extend or withhold
  opportunities, resources, or information. Some of the key applications are in
  hiring, school admissions, and lending.

* *Quality-of-service harms* can occur when a system does not work as well for
  one person as it does for another, even if no opportunities, resources, or
  information are extended or withheld. Examples include varying accuracy in
  face recognition, document search, or product recommendation.

Fairness assessment and unfairness mitigation
---------------------------------------------

In Fairlearn, we provide tools to assess fairness of predictors for
classification and regression. We also provide tools that mitigate unfairness
in classification and regression. In both assessment and mitigation scenarios,
fairness is quantified using disparity metrics as we describe below.

Group fairness, sensitive features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are many approaches to conceptualizing fairness. In Fairlearn, we follow
the approach known as group fairness, which asks: *Which groups of individuals
are at risk for experiencing harms?*

The relevant groups (also called subpopulations) are defined using **sensitive
features** (or sensitive attributes), which are passed to a Fairlearn
estimator as a vector or a matrix called :code:`sensitive_features` (even if it is
only one feature). The term suggests that the system designer should be
sensitive to these features when assessing group fairness. Although these
features may sometimes have privacy implications (e.g., gender or age) in
other cases they may not (e.g., whether or not someone is a native speaker of
a particular language). Moreover, the word sensitive does not imply that
these features should not be used to make predictions – indeed, in some cases
it may be better to include them.

Fairness literature also uses the term *protected attribute* in a similar
sense as sensitive feature. The term is based on anti-discrimination laws
that define specific *protected classes*. Since we seek to apply group
fairness in a wider range of settings, we avoid this term.

Parity constraints
^^^^^^^^^^^^^^^^^^

Group fairness is typically formalized by a set of constraints on the behavior
of the predictor called **parity constraints** (also called criteria). Parity
constraints require that some aspect (or aspects) of the predictor behavior be
comparable across the groups defined by sensitive features.

Let :math:`X` denote a feature vector used for predictions, :math:`A` be a
single sensitive feature (such as age or race), and :math:`Y` be the true
label. Parity constraints are phrased in terms of expectations with respect to
the distribution over :math:`(X,A,Y)`.
For example, in Fairlearn, we consider the following types of parity constraints.

*Binary classification*:

* *Demographic parity* (also known as *statistical parity*): A classifier
  :math:`h` satisfies demographic parity under a distribution over
  :math:`(X, A, Y)` if its prediction :math:`h(X)` is statistically
  independent of the sensitive feature :math:`A`. This is equivalent to
  :math:`\E[h(X) \given A=a] = \E[h(X)] \quad \forall a`. [#3]_

* *Equalized odds*: A classifier :math:`h` satisfies equalized odds under a
  distribution over :math:`(X, A, Y)` if its prediction :math:`h(X)` is
  conditionally independent of the sensitive feature :math:`A` given the label
  :math:`Y`. This is equivalent to
  :math:`\E[h(X) \given A=a, Y=y] = \E[h(X) \given Y=y] \quad \forall a, y`.
  [#3]_

* *Equal opportunity*: a relaxed version of equalized odds that only considers
  conditional expectations with respect to positive labels, i.e., :math:`Y=1`.
  [#2]_

*Regression*:

* *Demographic parity*: A predictor :math:`f` satisfies demographic parity
  under a distribution over :math:`(X, A, Y)` if :math:`f(X)` is independent
  of the sensitive feature :math:`A`. This is equivalent to
  :math:`\P[f(X) \geq z \given A=a] = \P[f(X) \geq z] \quad \forall a, z`.
  [#1]_

* *Bounded group loss*: A predictor :math:`f` satisfies bounded group loss at
  level :math:`\zeta` under a distribution over :math:`(X, A, Y)` if
  :math:`\E[loss(Y, f(X)) \given A=a] \leq \zeta \quad \forall a`. [#1]_

Above, demographic parity seeks to mitigate allocation harms, whereas bounded
group loss primarily seeks to mitigate quality-of-service harms. Equalized
odds and equal opportunity can be used as a diagnostic for both allocation
harms as well as quality-of-service harms.

Disparity metrics, group metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Disparity metrics evaluate how far a given predictor departs from satisfying a
parity constraint. They can either compare the behavior across different
groups in terms of ratios or in terms of differences. For example, for binary
classification:

* *Demographic parity difference* is defined as
  :math:`(\max_a \E[h(X) \given A=a]) - (\min_a \E[h(X) \given A=a])`.
* *Demographic parity ratio* is defined as
  :math:`\dfrac{\min_a \E[h(X) \given A=a]}{\max_a \E[h(X) \given A=a]}`.

The Fairlearn package provides the functionality to convert common accuracy
and error metrics from `scikit-learn` to *group metrics*, i.e., metrics that
are evaluated on the entire data set and also on each group individually.
Additionally, group metrics yield the minimum and maximum metric value and for
which groups these values were observed, as well as the difference and ratio
between the maximum and the minimum values. For more information refer to the
subpackage :code:`fairlearn.metrics`.

Failure modes encountered when abstracting
--------------------------------------------------

In "Fairness and Abstraction in Sociotechnical Systems" Selbst et al. [#4]_
identify failure modes that arise from applying abstraction. They identify
them as:

* *The Framing Trap*

* *The Portability Trap*

* *The Formalism Trap*

* *The Ripple Effect Trap*

* *The Solutionism Trap*

These traps are inspired by the observation that each of these traps are the
result of failing to consider the way in with social context and technology
are interrelated, as well as a deeper understanding of "the social" in order to
solve problems Ackerman [#5]_.

The Framing Trap
^^^^^^^^^^^^^^^^

Selbst et al. [#5]_ define this as a "failure to model the entire system over
which a social criterion, such as fairness, will be enforced".

It is the practice of failing to consider the full picture surrounding sources of
bias.

Labelling is a risk assessment setting. Risk assessment may be used as a predictive
model whose goal is to assess the risk of an individual to ease the decision-making
process. For example, using risk assessment to decide whether a person is eligible
for mortgage approval. Another example is using risk assessment to decide whether a
person is likely to re-engage in criminal behaviour as they move through the criminal
justice pipeline.

As far as the outcomes of these decisions go, in the first scenario an individual may be
approved or not to get a mortgage which then impacts their ability to have housing security,
gain wealth and social status. In the second scenario an individual may be released pretrial,
or bail may be required. This assessment may also include sentencing and parole hearing.
Risk assessment tools may be used as recommendations towards an outcome. However there is no
guarantee that they will be used consistently and thus bias is introduced. For example, a
party responsible for choosing the outcome decides to ignore the recommendation as a result
of personal bias, even when the model takes into account demographic data. In this example
a model that fails to take into account bias present in those judging culminates in a
biased result.

The Portability Trap
^^^^^^^^^^^^^^^^^^^^

Selbst et al. [#5]_ define this as a "failure to understand how repurposing algorithmic
solutions designed for one social context may be misleading, inaccurate, or otherwise
do harm when applied to a different context".

It is the practice of reusing others work without taking into account lack of
transferability to a given social situation, resulting in harmful consequences.

The Formalism Trap
^^^^^^^^^^^^^^^^^^

Selbst et al. [#5]_ define this as a "failure to account for the full meaning of social concepts
such as fairness, which can be procedural, contextual, and contestable, and cannot be resolved
through mathematical formalisms".

It is the practice of implementing mathematical and statistical models, along with
corresponding assumptions, that fail to take into consideration the social, demographic,
economic, or otherwise non-technical aspect that make up the phenomenon being studied.

The Ripple Effect Trap
^^^^^^^^^^^^^^^^^^^^^^

Selbst et al. [#5]_ define this as a "failure to understand how the insertion of technology into
an existing social system changes the behaviors and embedded values of the pre-existing system".


The Solutionism Trap
^^^^^^^^^^^^^^^^^^^^

Selbst et al. [#5]_ define this as a "failure to recognize the possibility that the best solution
to a problem may not involve technology".

It is the practice of assuming that a machine learning algorithm is the best solution to a problem.

One area where this manifests in contexts in which the definition "fairness" changes or is dependent
on a political context. When this happens, models become obsolete. Another area where this manifests
is when the question at hand requires a computationally intractable solution.

.. topic:: References:

   .. [#1] Agarwal, Dudik, Wu `"Fair Regression: Quantitative Definitions and
      Reduction-based Algorithms" <https://arxiv.org/pdf/1905.12843.pdf>`_,
      ICML, 2019.
   
   .. [#2] Hardt, Price, Srebro `"Equality of Opportunity in Supervised
      Learning"
      <https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf>`_,
      NIPS, 2016.
   
   .. [#3] Agarwal, Beygelzimer, Dudik, Langford, Wallach `"A Reductions
      Approach to Fair Classification"
      <https://arxiv.org/pdf/1803.02453.pdf>`_, ICML, 2018.
	  
   .. [#4] Selbst, Andrew D. and Boyd, Danah and Friedler, Sorelle and Venkatasubramanian,
        Suresh and Vertesi, Janet, "Fairness and Abstraction in Sociotechnical Systems" (August 23, 2018).
        2019 ACM Conference on Fairness, Accountability, and Transparency (FAT*), 59-68, Available at
        `SSRN: 	<https://ssrn.com/abstract=3265913>`_,
   
   .. [#5] Mark S. Ackerman. 2000. The intellectual challenge of CSCW: The gap between social requirements
        and technical feasibility. Human-Computer Interaction 15, 2-3 (2000), 179–203.
