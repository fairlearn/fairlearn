.. _how_to_talk_about_fairness:

How to talk and write about fairness in AI
------------------------------------------

The following style guide builds on work from various sources including the
Fairlearn contributors and
`Microsoft's Aether Fairness Working Group <https://www.microsoft.com/en-us/ai/responsible-ai>`_.
It is meant to provide a clear and easy to follow guide for contributors.
Every pull request is expected to abide by the guide. If you want to add to
the list feel free to send a pull request.

- Be clear that there is no single definition of fairness that will apply
  equally well to all AI systems.
- Be clear that any quantitative definition of fairness will omit aspects of
  fairness (as a societal concept) that cannot be quantified (e.g., justice,
  due process, remedying historical societal injustices).
- Be clear that given the many complex sources of unfairness, it is not
  possible to fully *debias* a system or to guarantee fairness. The goal is to
  assess and mitigate fairness-related harms as much as possible.
  For this reason, don’t usewords like *debias*, *unbiased*, *solve* – they
  set up unrealistic expectations. Use words like *mitigate*, *address*,
  *prioritize*, *assess* instead.
- Be clear that AI systems (and technology in general) are never *neutral* –
  all AI systems necessarily reflect the assumptions, priorities, and values
  of the people involved in their development and deployment.
- Be clear that prioritizing fairness in AI systems often means making
  tradeoffs based on competing priorities. There are seldom clear-cut answers.
  This is why using the word *solve* is seldom appropriate.
- Be clear that prioritizing fairness in AI systems is a sociotechnical
  challenge. It is not something that can be accomplished via purely technical
  methods (or purely social methods, for that matter).
- Be clear that there is no software tool that will *solve* fairness in all AI
  systems. This is not to say that software tools don’t have a role to play,
  but they will be precise, targeted, and only part of the picture.
- Be clear that even with precise, targeted tools, it’s easy to overlook
  things, especially things that are difficult to quantify – software tools
  must be supplemented with other resources and processes.
- There are many reasons why AI systems can behave unfairly, not just societal
  biases. Also *bias* is ambiguous and means different things to different
  communities – e.g., statistical bias vs. societal biases. For this reason,
  talk about *fairness issues* or *fairness-related harms* rather than
  *bias*, unless you are very specifically referring to societal biases
  (or statistical bias or some other definition of bias). Better yet, be
  specific about the type of fairness-related harm – is this a harm of
  allocation, quality of service, stereotyping, denigration, or over- or
  under-representation? Be clear that different types of fairness-related
  harm are not mutually exclusive. A single AI system can exhibit more than
  one type.
- Be clear that fairness-related harms can have varying severities, but that
  the cumulative impact of even *non-severe* harms can be extremely burdensome
  or make people feel singled out or undervalued.
- Be clear that fairness-related harms can affect both the people who will use
  an AI system and the people who will be directly or indirectly affected by
  the system, either by choice or not. For this reason, when talking about the
  people who might be harmed by a system, talk about *stakeholders* not
  *users*.
- When talking about who might be harmed, don’t just focus on demographic
  groups (e.g., groups defined in terms of race, gender, age, disability
  status, skin tone, and their intersections) or groups that are protected by
  anti-discrimination laws. The most relevant groups may be context specific.
- Be clear that stakeholders can belong to overlapping or intersectional
  groups – e.g., different combinations of race, gender, and age – and
  considering each group separately may obscure harms.
- Be clear that fairness-related harms can be (re-)introduced at every stage
  of the AI development and deployment lifecycle and can arise due to any
  component of an AI system, not just datasets or models – e.g., task
  definitions, user experiences, evaluation metrics, or deployment contexts.
