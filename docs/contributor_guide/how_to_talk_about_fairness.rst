.. _how_to_talk_about_fairness:

How to talk and/or write about fairness in AI
---------------------------------------------

The guidelines below represent learnings from commonly observed patterns.
We hope it may help contributors in communicating based on shared terminology
and with a consistent view on the topic of fairness. We expect the guide to
get updated regularly and all feedback is welcome.

To start, we need to define basic AI terminology:

- Talk about the *AI development and deployment lifecycle* rather than, e.g.,
  the *ML pipeline*.
- Talk about an *AI system* when you’re referring to something that might
  consist of more than one model.

With these definitions in mind we can approach the topic of fairness in AI.
From a high-level point of view we need to be be clear that

- AI systems (and technology in general) are never *neutral*. All AI systems
  necessarily reflect the assumptions, priorities, and values of the people
  involved in their development and deployment;
- fairness-related harms can be (re-)introduced at every stage
  of the AI development and deployment lifecycle and can arise due to any
  component of an AI system (e.g., task definitions, user experiences,
  evaluation metrics, or deployment contexts), not just datasets or models.

To complicate matters, there is no single definition of fairness that will
apply equally well to all AI systems. TODO: reference fairness and abstraction in sociotechnical systems
Any quantitative definition of fairness will omit aspects of fairness (as a
societal concept) that cannot be quantified (e.g., justice, due process,
remedying historical societal injustices). TODO: reference fairness and abstraction in sociotechnical systems
As a consequence, each application context needs to be examined individually
and we need to represent all considerations appropriately.

The choice of specific words in this context can make a difference. Since
there are many complex sources of unfairness, it is not possible to fully
*debias* a system or to guarantee fairness. The goal is to detect and to
mitigate fairness-related harms as much as possible. For this reason,

- don’t use words like *debias*, *unbiased*, *solve* – they set up
  unrealistic expectations;
- use words like *mitigate*, *address*, *prioritize*, *detect*, *identify*,
  *assess* instead;
- avoid the term *bias* unless you are very specifically referring to a kind
  of bias. *Bias* is an ambiguous term that means different things to
  different communities - e.g., statistical bias vs. societal biases. Since
  there are many reasons - not just societal biases - why AI system can behave
  unfairly, talk about *fairness issues* or *fairness-related harms* instead.
- using the word *solve* is seldom appropriate because prioritizing fairness
  in AI systems often means making tradeoffs based on competing priorities,
  with no clear-cut answers.

Prioritizing fairness in AI systems is a fundamentally sociotechnical
challenge. It cannot be accomplished via purely technical methods (or purely
social methods, for that matter). When discussing softare tools such as
Fairlearn we need to be clear that there is no software tool that will *solve*
fairness in all AI systems. This is not to say that software tools don’t have
a role to play, but they will be precise, targeted, and only part of the
picture. Even with precise, targeted tools, it’s easy to overlook things,
especially things that are difficult to quantify. That is why software tools
must be supplemented with other resources and processes.

In the context of fairness-related harms be specific about the types of harms:
is it a harm of

* allocation,
* quality of service,
* stereotyping,
* denigration, or
* over- or under-representation?

For more on different types of harms refer to
:ref:`fairness_in_machine_learning`.

Note that harms are not mutually exclusive. Furthermore, they can have varying
severities, but the cumulative impact of even *non-severe* harms can be
extremely burdensome or make people feel singled out or undervalued. They can
affect both the people who will use an AI system and the people who will be
directly or indirectly affected by the system, either by choice or not. For
this reason, when talking about the people who might be harmed by a system,
talk about *stakeholders*, not *users*. When talking about stakeholders, don’t
just focus on demographic groups (e.g., groups defined in terms of race,
gender, age, disability status, skin tone, and their intersections) or groups
that are protected by anti-discrimination laws. The most relevant groups may
be context specific. Additionally, stakeholders can belong to overlapping or
intersectional groups – e.g., different combinations of race, gender, and
age – and considering each group separately may obscure harms.
