.. _fairness_in_machine_learning:
.. _terminology:

Fairness in Machine Learning
============================

Fairness of AI systems
----------------------

AI systems can behave unfairly for a variety of reasons. Sometimes it is
because of societal biases reflected in the training data and in the decisions
made during the development and deployment of these systems. In other cases, AI systems behave unfairly due to characteristics of the data (e.g., too few data points about some group of
people) or characteristics of the systems themselves. It can be hard to
distinguish between these reasons, especially since they are not mutually
exclusive and often exacerbate one another. Therefore, we define whether an AI
system is behaving unfairly in terms of its impact on people — i.e., in terms
of harms — and not in terms of specific causes, such as societal biases, or in
terms of intent, such as prejudice.

**Usage of the word bias.** Since we define fairness in terms of harms
rather than specific causes (such as societal biases), we avoid the usage of
the words *bias* or *debiasing* in describing the functionality of Fairlearn.

.. _types_of_harms:

Types of harms
--------------

There are many types of harms :footcite:`barocas2017problem`.
Some of these are:

* *Allocation harms* can occur when AI systems extend or withhold
  opportunities, resources, or information. Some of the key applications are in
  hiring, school admissions, and lending.

* *Quality-of-service harms* can occur when a system does not work as well for
  one person as it does for another, even if no opportunities, resources, or
  information are extended or withheld. Examples include varying accuracy in
  face recognition, document search, or product recommendation.

* *Stereotyping harms* can occur when a system suggests completions which
  perpetuate stereotypes.
  These are often seen when search engines propose completions to partially
  typed queries.
  See :footcite:cts:`umojanoble2018algorithmsoppression` for an in-depth
  look at this issue.
  Note that even stereotypes which are nominally positive are also
  problematic, since they still create expectations based on outward
  characteristics, rather than treating people as individuals.

* *Erasure harms* can occur when a system behaves as if groups (or their
  works) do not exist.
  For example, a text generator prompted about "Female scientists of the 1800s"
  might not produce a result.
  When asked about historical sites near St. Louis, Missouri, a search engine
  might fail to mention `Cahokia <https://en.wikipedia.org/wiki/Cahokia>`_.
  A similar query about southern Africa might overlook
  `Great Zimbabwe <https://en.wikipedia.org/wiki/Great_Zimbabwe>`_, instead
  concentrating on colonial era sites.
  More subtly, a short biography of Alan Turing might not mention his
  sexuality.

This list is not exhaustive, and it is important to remember that harms
are not mutually exclusive.
A system can harm multiple groups of people in different ways, and also
visit multiple harms on a single group of people.
The Fairlearn package is most applicable to allocation and quality of service harms,
since these are easiest to measure.

.. _concept_glossary:

Concept glossary
----------------------------

The concepts outlined in this glossary are relevant to sociotechnical contexts.

Construct validity
^^^^^^^^^^^^^^^^^^
In many cases, fairness-related harms can be traced back to the way a real-world problem is translated into a machine learning task.
Which target variable do we intend to predict?
What features will be included?
What (fairness) constraints do we consider?
Many of these decisions boil down to what social scientists refer to as measurement: the way we measure (abstract) phenomena.

The concepts outlined in this glossary give an introduction into the language of measurement modeling - as described in :footcite:cts:`jacobs2021measurement`.
This framework can be a useful tool to test the validity of (implicit) assumptions of a problem formulation.
In this way, it can help to mitigate fairness-related harms that can arise from mismatches between the formulation and the real-world context of an application.

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

- **Patient benefit** is an example of an unobservable theoretical construct. A common measurement model involves patient care costs. See :footcite:`obermeyer2019dissecting` for a related example.

**Note:**
We cite several examples of unobservable theoretical constructs and measurement models for the purpose of explaining the key terms outlined above.
Please reference :footcite:cts:`jacobs2021measurement` for more detailed examples.


.. _construct_validity:

What is construct validity?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Though :footcite:cts:`jacobs2021measurement` explore both construct reliability and construct validity, we focus our
exploration below on construct Validity.
We note that both play an important role in understanding fairness in sociotechnical contexts.
With that said, :footcite:cts:`jacobs2021measurement` offers a fairness-oriented conceptualization of construct validity, that
is helpful in thinking about fairness in sociotechnical contexts.
We capture the idea in seven key parts that when combined  can serve as a framework for analyzing an AI task and attempting to establish construct validity:

1. **Face validity** – On the surface, how plausible do the measurements produced by the measurement model look?

2. **Content validity** – This has three subcomponents:

   a. **Contestedness** – Is there a single understanding of the unobservable theoretical construct? Or is that understanding contested (and thus context dependent).
   b. **Substantive validity** – Can we demonstrate that the measurement model contains the observable properties and other unobservable theoretical constructs related to the construct of interest (and only those)?
   c. **Structural validity** – Does the measurement model appropriately capture the relationships between the construct of interest and the measured observable properties and other unobservable theoretical constructs?

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


.. _parity_constraints:

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
  :math:`\E[h(X) \given A=a] = \E[h(X)] \quad \forall a`.  :footcite:`agarwal2018reductions`

* *Equalized odds*: A classifier :math:`h` satisfies equalized odds under a
  distribution over :math:`(X, A, Y)` if its prediction :math:`h(X)` is
  conditionally independent of the sensitive feature :math:`A` given the label
  :math:`Y`. This is equivalent to
  :math:`\E[h(X) \given A=a, Y=y] = \E[h(X) \given Y=y] \quad \forall a, y`.
  :footcite:`agarwal2018reductions`

* *Equal opportunity*: a relaxed version of equalized odds that only considers
  conditional expectations with respect to positive labels, i.e., :math:`Y=1`.
  :footcite:`hardt2016equality`

*Regression*:

* *Demographic parity*: A predictor :math:`f` satisfies demographic parity
  under a distribution over :math:`(X, A, Y)` if :math:`f(X)` is independent
  of the sensitive feature :math:`A`. This is equivalent to
  :math:`\P[f(X) \geq z \given A=a] = \P[f(X) \geq z] \quad \forall a, z`.
  :footcite:`agarwal2019fair`

* *Bounded group loss*: A predictor :math:`f` satisfies bounded group loss at
  level :math:`\zeta` under a distribution over :math:`(X, A, Y)` if
  :math:`\E[loss(Y, f(X)) \given A=a] \leq \zeta \quad \forall a`. :footcite:`agarwal2019fair`

Above, demographic parity seeks to mitigate allocation harms, whereas bounded
group loss primarily seeks to mitigate quality-of-service harms. Equalized
odds and equal opportunity can be used as a diagnostic for both allocation
harms as well as quality-of-service harms.


.. _disparity_metrics:

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



.. _abstraction_traps:

What traps can we fall into when modeling a social problem?
--------------------------------------------------------------

Machine learning systems used in the real world are inherently sociotechnical
systems, which include both technologies and social actors. Designers of machine
learning systems typically translate a real-world context into a machine learning
model through abstraction: focusing only on 'relevant' aspects of that context,
which are typically described by inputs, outputs, and the relationship between them.
However, by abstracting away the social context they are at risk of falling into
'abstraction traps': a failure to consider how social context and technology
are interrelated.

In this section, we explain what those traps are, and give some suggestions on
how we can avoid them.

In "Fairness and Abstraction in Sociotechnical Systems," :footcite:ct:`selbst2019fairness`
identify failure modes that can arise from abstracting away the social context
when modeling. They identify them as:

* *The Solutionism Trap*

* *The Ripple Effect Trap*

* *The Formalism Trap*

* *The Portability Trap*

* *The Framing Trap*

We provide some definitions and examples of these traps to help Fairlearn
users think about how choices they make in their work can lead to or avoid
these common pitfalls.


.. _solutionism_trap:

The Solutionism Trap
^^^^^^^^^^^^^^^^^^^^

This trap occurs when we assume that the best solution to a problem
may involve technology, and fail to recognize other possible solutions
outside of this realm. Solutionist approaches may also not be appropriate
in situations where definitions of fairness may change over time
(see 'The Formalism Trap' below).

Example: consider the problem of internet connectivity in rural communities.
An example of the solutionism trap is assuming that using data science to
measure internet speed in a given region
can help improve internet connectivity.
However, if there are additional socioeconomic challenges within
a community, for example with education, infrastructure, information
technology, or health services, then an algorithmic solution purely
focused on internet speed may fail to meaningfully address the needs of
the community.


.. _ripple_effect_trap:

The Ripple Effect Trap
^^^^^^^^^^^^^^^^^^^^^^

This trap occurs when we do not consider the unintended consequences of
introducing technology into an existing social system. Such consequences
include changes in behaviors, outcomes, individual experiences, or changes
in underlying social values and incentives of a given social system; for
instance, by increasing perceived value of quantifiable metrics over
non-quantifiable ones.

Example: consider the problem of banks deciding whether an individual should
be approved for a loan. Prior to using machine learning algorithms
to compute a "score", banks might rely on loan officers that engage in
conversations with clients, recommend a plan based on their unique
situation, and discuss with other team members to obtain feedback.
By introducing an algorithm, it is possible that loan officers may limit
their conversations with team members and clients, assuming the algorithm's
recommendations are good enough without those additional sources of information.

To avoid this pitfall, we must be aware that once a technology is incorporated
into a social context, new groups may reinterpret it differently. We should
adopt "what if" scenarios to envision how the social context might change
after introducing a model, including how it may change the power dynamics of
existing groups in that context, or how actors might change their behaviors to
game the model.


.. _formalism_trap:

The Formalism Trap
^^^^^^^^^^^^^^^^^^

Many tasks of a data scientist involve some form of formalization: from
measuring real-world phenomena as data to translating business Key Performance
Indicators (KPIs) and constraints into metrics, loss functions, or parameters.
We fall into the formalism trap when we fail to account for the full meaning
of social concepts like fairness.

Fairness is a complex construct that is contested: different people may
have different ideas of what is fair in a particular scenario. While
mathematical fairness metrics may capture some aspects of fairness, they
fail to capture all relevant aspects. For example, group fairness metrics
do not account for differences in individual experiences, nor do they
account for procedural justice.

In some scenarios, fairness metrics such as demographic parity and equalized
odds cannot be satisfied at the same time. At a first glance, this may appear
to be a mathematical problem. However, the conflict is actually grounded in
different understandings of what fairness is. Consequently, there is no
mathematical approach to solve the conflict. Instead we need to decide which
metrics might be appropriate for the situation at hand, keeping in mind the
limitations of a mathematical formalization. In some cases, there may be no
suitable metric.

Some reasons why we fall into this trap are because fairness is
context-dependent, because it is open to contestation by different groups
of people, and because there are differences between ways of thinking about
fairness between the legal world (i.e., fairness as procedural) and the fair-ML
community (i.e., fairness as outcome-based).

Where mathematical abstraction encounters a limitation is when
capturing information regarding contextuality (different communities
may have different definitions for what constitutes an "unfair" outcome;
for instance, is it unfair to hire an applicant whose primary language
is English, for an English speaking role, over an applicant whose only
spoken language is not English?); contestability (the definitions of
discrimination and unfairness are politically contested and change
over time, which may pose fundamental challenges for representing
them mathematically); and procedurality (for example, how do judges
and police officers determine whether bail, counselling, probation, or
incarceration is appropriate);


.. _portability_trap:

The Portability Trap
^^^^^^^^^^^^^^^^^^^^

This trap occurs when we fail to understand how reusing a model or
algorithm that is designed for one specific social context may not
necessarily apply to a different social context. Reusing an algorithmic
solution and failing to take into account differences in involved social
contexts can result in misleading results and potentially harmful consequences.

For instance, reusing a machine learning algorithm used to screen
job applications in the nursing industry for a system used to screen
job applications in the information technology sector could fall into the
portability trap. One important difference between both contexts is
the difference in skills required to succeed in both industries.
Another key difference between these contexts involves the demographic
differences (in terms of gender) of employees in each of these industries,
which may result from wording in job postings, social constructs on gender
and societal roles, and the percentages of successful applicants in
each field per (gender) group.


.. _framing_trap:

The Framing Trap
^^^^^^^^^^^^^^^^

This trap occurs when we fail to consider the full picture surrounding
a particular social context when abstracting a social problem. Elements
involved include but are not limited to: the social landscape that the
chosen phenomenon exists in, characteristics of individuals or
circumstances of the chosen situation, third parties involved along with
their circumstances, and the task that is being set out to abstract
(i.e., calculating a risk score, choosing between a pool of candidates,
selecting an appropriate treatment, etc).

To help us avoid drawing narrow boundaries of what is considered in scope
for the problem, we might consider using wider "frames" around what is
considered to be in scope for the problem, moving from an algorithmic frame
to a sociotechnical frame.

For instance, adopting a *sociotechnical* frame (instead of a data-focused,
or algorithmic frame) allows us to recognize that a machine learning model
is part of social and technical interactions between people and technology,
and thus the social components of a given social context should be included
as part of the problem formulation and modeling approach (including local
decision-making processes, incentive structures, institutional processes,
and more).

For instance, we might fall into this trap by assessing risk of re-engagement
in criminal behavior for an individual charged with an offense, while failing
to consider factors such as the legacy of racial biases in criminal justice
systems, the relationship of socio-economic status and mental health to the
social construction of criminality, along with existing societal biases of
judges, police officers, or other social actors involved in the larger
sociotechnical frame around a criminal justice algorithm.

Within the sociotechnical frame the model incorporates not only more
nuanced data on the history of the case, but also the social context in
which judging and recommending an outcome take place. This frame might
incorporate the processes associated with crime reporting, the offense-trial
pipeline, and an awareness of how the relationship between various social actors and
the algorithm may impact the intended outcomes of a given model.

Stakeholder Identification
--------------------------
Introduction
^^^^^^^^^^^^

Now that we've seen how AI systems can generate harms, what else should we account for when designing an AI system?
AI systems impact not only end users but also organizations within a business, communities, civil society, government agencies, and entire industries. If practitioners do not have a process to engage stakeholders, they may rely on their personal experiences and identities instead and overlook fairness-related harms :footcite:`madaio2022assess`.  Therefore, it is important to identify the relevant stakeholders, factors, and groups that might be at the most risk of experiencing fairness-related harms before conducting a fairness assessment.

.. _defining_terms:
Defining terms
^^^^^^^^^^^^^^
**Stakeholders** include direct stakeholders, people that use or operate an AI system, or indirect stakeholders (people that could be harmed by a system that are not necessarily users or customers) :footcite:`madaio2022assess`. For example, in the case of a fraud detection AI system, there could be three types of stakeholders identified: a) people whose transactions might be mistakenly classified as fraudulent, b) companies running money-transfer platforms with the fraud detection AI system, and c) local government fraud auditors that audit the money transfer platforms' transactions.

Note: Some have identified that using the term *stakeholder* may perpetuate colonial harm in some contexts :footcite:`reed2024stakeholder`. It is important to pay attention to the context in which stakeholder identification occurs and adjust accordingly (e.g. opt for alternative terminology to avoid causing harm).

**Factors and groups** include not only demographic factors (e.g. race, gender, age) but also sociocultural factors (e.g., head coverings, facial hair, glasses), behavioral factors (e.g., walking speed) and morphological (e.g., body shape, skin tone) :footcite:`barocas2021disagg` .

References
----------

.. footbibliography::
