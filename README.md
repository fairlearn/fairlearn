[![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Nightly?branchName=master)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=23&branchName=master) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![PyPI](https://img.shields.io/pypi/v/fairlearn?color=blue) [![Gitter](https://badges.gitter.im/fairlearn/community.svg)](https://gitter.im/fairlearn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![StackOverflow](https://img.shields.io/badge/StackOverflow-questions-blueviolet)](https://stackoverflow.com/questions/tagged/fairlearn)

# Fairlearn - Alpha release

Fairlearn is an open source toolkit that empowers data scientists and developers to assess and improve the fairness of their AI systems.  Fairlearn has three components: Python functions for computing fairness metrics, mitigation algorithms for optimizing fairness metrics, and an interactive visualization dashboard.

While these components are designed to help with navigating trade-offs between fairness and model performance, we emphasize that prioritizing fairness in AI systems is a sociotechnical challenge.  Because there are many complex sources of unfairness — some societal and some technical — it is not possible to fully “debias” a system or to guarantee fairness; the goal is to mitigate fairness-related harms as much as possible.

Read more in [Fairlearn: A toolkit for assessing and improving fairness in AI](https://www.microsoft.com/en-us/research/uploads/prod/2020/05/Fairlearn_whitepaper.pdf) or at [fairlearn.github.io](https://fairlearn.github.io/).

- [Current release](#current-release)
- [What we mean by _fairness_](#what-we-mean-by-fairness)
- [Overview of Fairlearn](#overview-of-fairlearn)
  - [Fairlearn algorithms](#fairlearn-algorithms)
  - [Fairlearn dashboard](#fairlearn-dashboard)
- [Install Fairlearn](#install-fairlearn)
- [Usage](#usage)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [Issues](#issues)

## Current release

The current stable release is available at [Fairlearn v0.4.6](https://github.com/fairlearn/fairlearn/tree/v0.4.6).  For older versions, see the [changelog](CHANGES.md) or [upgrading guide](https://fairlearn.github.io/user_guide/upgrading.html).

Fairlearn is currently an **alpha** project.  It started as a Python package to accompany the research paper, "A Reductions Approach to Fair Classification," ([Agarwal et al. 2018](https://arxiv.org/abs/1803.02453)), and we're working to translate that research into an open source toolkit that empowers data scientists and developers to assess and improve the fairness of their AI systems.



## What we mean by _fairness_

An AI system can behave unfairly for a variety of reasons. In Fairlearn, we define whether an AI system is behaving unfairly in terms of its impact on people &ndash; i.e., in terms of harms. We focus on two kinds of harms:

- _Allocation harms._ These harms can occur when AI systems extend or withhold opportunities, resources, or information. Some of the key applications are in hiring, school admissions, and lending.

- _Quality-of-service harms._ Quality of service refers to whether a system works as well for one person as it does for another, even if no opportunities, resources, or information are extended or withheld.

We follow the approach known as **group fairness**, which asks: _Which groups of individuals are at risk for experiencing harms?_ The relevant groups need to be specified by the data scientist and are application specific.

Group fairness is formalized by a set of constraints, which require that some aspect (or aspects) of the AI system's behavior be comparable across the groups. The Fairlearn package enables assessment and mitigation of unfairness under several common definitions.
To learn more about our definitions of fairness, please visit our
[user guide on Fairness of AI Systems](https://fairlearn.github.io/user_guide/fairness_in_machine_learning.html#fairness-of-ai-systems).

>_Note_:
> Fairness is fundamentally a sociotechnical challenge. Many aspects of fairness, such as justice and due process, are not captured by quantitative fairness metrics. Furthermore, there are many quantitative fairness metrics which cannot all be satisfied simultaneously. Our goal is to enable humans to assess different mitigation strategies and then make trade-offs appropriate to their scenario.

## Overview of Fairlearn

The Fairlearn Python package has two components:

- A _dashboard_ for assessing which groups are negatively impacted by a model, and for comparing multiple models in terms of various fairness and accuracy metrics.

- _Algorithms_ for mitigating unfairness in a variety of AI tasks and along a variety of fairness definitions.

### Fairlearn algorithms

For an overview of our algorithms please refer to our [website](https://fairlearn.github.io/user_guide/mitigation.html).

### Fairlearn dashboard

Check out our in-depth [guide on the Fairlearn dashboard](https://fairlearn.github.io/user_guide/assessment.html#fairlearn-dashboard).

## Install Fairlearn

For instructions on how to install Fairlearn check out our [Quickstart guide](https://fairlearn.github.io/quickstart.html).

## Usage

For common usage refer to the [Jupyter notebooks](./notebooks) and our
[user guide](https://fairlearn.github.io/user_guide/index.html).
Please note that our APIs are subject to change, so notebooks downloaded
from `master` may not be compatible with Fairlearn installed with `pip`.
In this case, please navigate the tags in the repository
(e.g. [v0.4.5](https://github.com/fairlearn/fairlearn/tree/v0.4.5))
to locate the appropriate version of the notebook.

## Contributing

To contribute please check our
[contributor guide](https://fairlearn.github.io/contributor_guide/index.html).

## Maintainers

The Fairlearn project is maintained by:

- **@MiroDudik**
- **@riedgar-ms**
- **@rihorn2**
- **@romanlutz**

For a full list of contributors refer to the [authors page](./AUTHORS.md)

## Issues

### Usage Questions

Pose questions and help answer them on [Stack
Overflow](https://stackoverflow.com/questions/tagged/fairlearn) with the tag
`fairlearn` or on [Gitter](https://gitter.im/fairlearn/community#).

### Regular (non-security) issues

Issues are meant for bugs, feature requests, and documentation improvements.
Please submit a report through
[GitHub issues](https://github.com/fairlearn/fairlearn/issues). A maintainer
will respond promptly as appropriate.

Maintainers will try to link duplicate issues when possible.

### Reporting security issues

Please take a look at our guidelines for reporting [security issues](./SECURITY.md).
