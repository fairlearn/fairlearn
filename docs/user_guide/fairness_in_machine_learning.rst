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

What traps can we fall into when modeling a social problem?
--------------------------------------------------------------

Machine learning systems used in the real world are inherently sociotechnical
systems, which include both technologies and social actors. Designers of machine
learning systems typically translate a real-world context into a machine learning
model through abstraction: focusing only on 'relevant' aspects of that context,
which are typically described by inputs, outputs, and the relationship between them.
However, by abstracting away the social context they are at risk of falling into
'abstraction traps': a failure to consider how social context and technology are interrelated.

In this section, we explain what those traps are, and give some suggestions on
how we can avoid them.

In "Fairness and Abstraction in Sociotechnical Systems," Selbst et al. [#4]_
identify failure modes that can arise from abstracting away the social context
when modeling. They identify them as:

* *The Solutionism Trap*

* *The Ripple Effect Trap*

* *The Formalism Trap*

* *The Portability Trap*

* *The Framing Trap*

We provide some definitions and examples of these traps to help Fairlearn
users think about how choices they make in their work can lead to or avoid these common pitfalls.

The Solutionism Trap
^^^^^^^^^^^^^^^^^^^^

This trap occurs when we assume that the best solution to a problem
may involve technology, and fail to recognize other possible solutions outside of
this realm.  Solutionist approaches may also not be appropriate in situations
where definitions of fairness may change over time (see the Formalism Trap below for more).
See also the "Construct Validity" section in Fairlearn's user guide.

Example: consider the problem of internet connectivity in rural communities.
An example of the solutionism trap is assuming that by using data science to
study internet speed in a given region, insights we gain from using data science
can help us in negotiating deals or discovering potential for good policies.
However, if there are additional socioeconomic problems within a community,
for example lack of education, infrastructure, information
technology and health services, then an algorithmic solution purely focused on internet
speed will fail to meaningfully address the needs of the community.

The Ripple Effect Trap
^^^^^^^^^^^^^^^^^^^^^^

This trap occurs when we do not consider the unintended consequences of introducing
technology into an existing social system. Such consequences include changes in
behavior, outcomes individual experience and a change in underlying social values
and incentives of a given social system, for instance by increasing perceived value
of quantifiable metrics over non-quantifiable ones.

Example: consider the problem of banks deciding whether an individual should
be approved for a loan. Prior to using machine learning algorithms
to compute a "score", banks might rely on loan officers engaging in conversations with
clients, recommending a plan based on their unique situation, and
discussing with other team members to obtain feedback. By introducing an
algorithm, it is possible that loan officers stop engaging in conversations
with team members and clients, and assume the algorithm is good enough
to accept its recommendation without those additional sources of information. 
information on people the algorithm typically rejects to create a system
that takes advantage of people who need funds but did not receive them.

To avoid this pitfall, we must be aware that once a technology is incorporated
into a social context, new groups may reinterpret it differently. We should
adopt "what if" scenarios to envision how the social context might change
after introducing a model, including how it may change the power dynamics of
existing groups in that context, or how actors might change their behaviors to
game the model.

The Formalism Trap
^^^^^^^^^^^^^^^^^^

Many tasks of a data scientist involve some form of formalization: from
measuring real-world phenomena as data to translating business KPI's
and constraints into metrics, loss functions, or parameters. We fall into the
formalism trap when we fail to account for the full meaning of social
concepts like fairness. This occurs because there is no purely mathematical way to resolve
conflicting definitions of fairness. This is also because fairness is complex and
contested by social actors, and it cannot only be captured mathematically, but
needs to be understood procedurally (i.e., [insert explanation here]) and situated in social contexts.

Because different definitions of fairness cannot be satisfied at the same time,
we'll need to decide which definition to use. But there's no mathematical way to
make that decision. And it might be that none of the definitions are appropriate for our situation.

Some reasons why we fall into this trap is because fairness is context-dependent,
open to contestation by different groups of people, and differences between ways of thinking
about fairness between the legal world (i.e., fairness as procedural)
and the fair-ML community (i.e., fairness as outcome-based).

Kleinberg et al. [#6]_ abstract the problem risk assessment via the use of
vectors to represent information about  person, boolean values to
represent group belonging, and risk assignment scores.

Where mathematical abstraction encounters a limitation is when capturing
information regarding procedurality (for example, how do judges and police officers
determine whether bail, counselling, probation, or incarceration is appropriate);
contextuality (different societies determine what constitutes an "unfair" outcome, furthermore
different groups determine what constitutes immoral discrimination, i.e. is it immoral
to hire an applicant whose primary language is not English, for a non-English speaking role, over
an applicant whose only spoken language is English?); and contestability (the definitions
of discrimination and unfairness are politically contested and change over time, 
which may pose fundamental challenges for representing them mathematically).

The Portability Trap
^^^^^^^^^^^^^^^^^^^^

This trap occurs when we fail to understand how reusing a model or
algorithm that is designed for one specific social context, may not necessarily
apply to a different social context. Reusing an algorithmic solution and failing
to take into account differences in involved social contexts can result in misleading
results and potentially harmful consequences. 

Example: Reusing a machine learning algorithm used to screen job applications in the
nursing industry, for job applications in the information technology sector. An intuitive
yet important difference between both contexts is the difference in skills required to
succeed in both industries. A slightly more subtle difference is the demographic differences
(in terms of gender) of employees in each of these industries, which may result from
wording in job postings, social constructs on gender and societal roles, and the male-female
ratio of successful applicants in each field.

The Framing Trap
^^^^^^^^^^^^^^^^

This trap occurs when we fail to consider the full picture surrounding
a particular social context when abstracting a social problem. Elements
involved include but are not limited to: the social landscape that the
chosen phenomenon exists in, characteristics of individuals or circumstances
of the chosen situation, third parties involved along with their circumstances,
and the task that is being set out to abstract (i.e. calculating a risk score,
choosing between a pool of candidates, selecting an appropriate treatment, etc).

To help us avoid drawing narrow boundaries of what is considered in scope for
the problem, we might consider using wider "frames" around what is considered to be in
scope for the problem, moving from an algorithmic frame to a sociotechnical frame.

The sociotechnical frame recognizes that a machine learning model is part
of social and technical interactions between people and technology, and thus the social
components of this within this social context should be included as part of the problem
and modeling approach (including local decision-making processes, incentive structures,
institutional processes, and more).

Example: assessing risk of re-engagement in criminal behaviour in an individual
charged with an offense, while failing
to consider factors such as race, socio-economic status, mental health, along with
socially-dependent views present in judges, police officers, or any actors responsible
for recommending a course of action.

In the algorithmic framework, for example, input variables may contain previous criminal history,
statements taken by the accused, witnesses and police officers. Labels (outcome)
include recommendations by the algorithm on an appropriate course of action based
on a computed risk score. Model is limited in assessing fairness out outcome.

The data framework could attempt to reduce unfairness by studying socio-economic
information regarding the accused, their upbringing and how it relates to their
current status, along with a recommendation that incorporates these factors into the
recommended outcome.

Within the sociotechnical framework the model incorporates not only more nuanced
data on the history of the case, but also the social context in which judging and
recommending an outcome take place. This frame incorporates the processes
associated with crime reporting, the offense-trial pipeline, and identifies areas
in which different people interact with one another as outcomes are recommended.

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
   
   .. [#5] Mark S. Ackerman. 2000. `"The intellectual challenge of CSCW: The gap between social requirements
      and technical feasibility" https://doi.org/10.1207/S15327051HCI1523_5`. Human-Computer
      Interaction 15, 2-3 (2000), 179–203.

   .. [#6] Jon Kleinberg, Sendhil Mullainathan, and Manish Raghavan. 2017. `"Inherent trade-offs
      in the fair determination of risk scores" https://arxiv.org/abs/1609.05807`.
      In Proc. of ITCS.

   .. [#7] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan,
      Hanna Wallach, Hal Daumé III, Kate Crawford. "Datasheets for Datasets"
      `https://arxiv.org/abs/1803.09010 <https://arxiv.org/abs/1803.09010>`_
      (2018)
