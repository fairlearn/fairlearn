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

Concept glossary
----------------------------

The concepts outlined in this glossary are relevant to sociotechnical contexts. 

Construct validity
^^^^^^^^^^^^^^^^^^
In many cases, fairness-related harms can be traced back to the way a real-world problem is translated into a machine learning task. Which target variable do we intend to predict? What features will be included? What (fairness) constraints do we consider? Many of these decisions boil down to what social scientists refer to as measurement: the way we measure (abstract) phenomena.

The concepts outlined in this glossary give an introduction into the language of measurement modeling - as described in Measurement and Fairness [#4]_ . This framework can be a useful tool to test the validity of (implicit) assumptions of a problem formulation. In this way, it can help to mitigate fairness-related harms that can arise from mismatches between the formulation and the real-world context of an application.

Key Terms 
~~~~~~~~~

- **Sociotechnical context** – The context surrounding a technical system, including both social aspects (e.g., people, institutions, communities) and technical aspects (e.g., algorithms, technical processes). The sociotechnical context of a system shapes who might benefit or is harmed by AI systems.

- **Unobservable theoretical construct** – An idea or concept that is unobservable and cannot be directly measured but must instead be inferred through observable measurements defined in a measurement model. 

- **Measurement model** – The method and approach used to measure the unobservable theoretical construct.

- **Construct reliability** – This can be thought of as the extent to which the measurements of an unobservable theoretical construct remain the same when measured at different points in time. A lack of construct reliability can either be due to a misalignment between the understanding of the unobservable theoretical construct and the methods being used to measure that construct, or to changes to the construct itself. Construct validity and construct reliability are complementary.

- **Construct validity** – This can be thought of as the extent to which the measurement model measures the intended construct in way that is meaningful and useful.

Key Term Examples  - Unobservable theoretical constructs and Measurement models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Fairness** is an example of an unobservable theoretical construct. Several measurement models exist for measuring fairness, including demographic parity. These measurements may come together to form a measurement model, where several measurements are combined to ultimately measure fairness.See :code:`fairlearn.metrics` for more examples of measurement models for measuring fairness.

- **Teacher effectiveness** is an example of an unobservable theoretical construct. Common measurement models include student performance on standardized exams and qualitative feedback for the teacher’s students.

- **Socioeconomic status** is an example of an unobservable theoretical construct. A common measurement model includes annual household income. 

- **Patient benefit** is an example of an unobservable theoretical construct. A common measurement model involves patient care costs. See [#5]_ for a related example. 

**Note:** We cite several examples of unobservable theoretical constructs and measurement models for the purpose of explaining the key terms outlined above.  Please reference Measurement and Fairness [#4]_ for more detailed examples.
 
What is construct validity? 
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Though the Measurement and Fairness [#4]_ paper explores both construct reliability and construct validity, we focus our 
exploration below on construct Validity. We note that both play an important role in understanding fairness in sociotechnical 
contexts. With that said, Measurement and Fairness [#4]_ offers a fairness-oriented conceptualization of construct validity, that 
is helpful in thinking about fairness in sociotechnical contexts. We capture the idea in seven key parts that when combined 
can serve as a framework for analyzing an AI task and attempting to establish construct validity:

1. **Face validity** – On the surface, how plausible do the measurements produced by the measurement model look?

2. **Content validity** – This has three subcomponents:

   a. **Contestedness** – Is there a single understanding of the unobservable theoretical construct? Or is that understanding contested (and thus context        dependent).
   b. **Substantive validity** – Can we demonstrate that the measurement model contains the observable properties and other unobservable 			   theoretical constructs related to the construct of interest (and only those)?
   c. **Structural validity** – Does the measurement model appropriately capture the relationships between the construct of interest and the measured           observable properties and other unobservable theoretical constructs?

3. **Convergent validity** – Do the measurements obtained correlate with other measurements (that exist) from 
   measurement models for which construct validity has been established? 
 
4. **Discriminant validity** – Do the measurements obtained for the construct of interest correlate with 
   related constructs as appropriate?  

5. **Predictive validity** – Are the measurements obtained from the measurement model predictive of measurements 
   of any relevant observable properties or other unobservable theoretical constructs?

6. **Hypothesis validity** – This describes the nature of the hypotheses that could emerge from the measurements 
   produced by the measurement model, and whether those are “substantively interesting”.

7. **Consequential validity** – Identify and evaluate the consequences and societal impacts of using the 
   measurements obtained for the measurement model. Framed as questions: how is the world shaped by using the 
   measurements, and what world do we wish to live in?

**Note:** The order in which the parts above are explored and the way in which they are used may vary depending on the specific 
sociotechnical context. This is only intended to explain the key concepts that could be used in a 
framework for analyzing a task.

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
subpackage :mod:`fairlearn.metrics`.


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

   .. [#4] Jacobs, Wallach `"Measurement and Fairness"
      <https://arxiv.org/pdf/1912.05511.pdf>`_, FAccT, 2021.

   .. [#5] Obermeyer, Powers, Vogeli, Mullainathan `"Dissecting racial bias in an algorithm used to manage the health of populations"
      <https://science.sciencemag.org/content/366/6464/447>`_, Science, 2019.
