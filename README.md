[![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Nightly?branchName=main)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=23&branchName=main) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![PyPI](https://img.shields.io/pypi/v/fairlearn?color=blue) [![Gitter](https://badges.gitter.im/fairlearn/community.svg)](https://gitter.im/fairlearn/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) [![StackOverflow](https://img.shields.io/badge/StackOverflow-questions-blueviolet)](https://stackoverflow.com/questions/tagged/fairlearn)

# Fairlearn

Fairlearn is a Python package that empowers developers of artificial intelligence (AI) systems to assess their system's fairness and mitigate any observed unfairness issues. Fairlearn contains mitigation algorithms as well as metrics for model assessment. Besides the source code, this repository also contains Jupyter notebooks with examples of Fairlearn usage.

Website: https://fairlearn.org/

- [Current release](#current-release)
- [What we mean by _fairness_](#what-we-mean-by-fairness)
- [Overview of Fairlearn](#overview-of-fairlearn)
  - [Fairlearn metrics](#fairlearn-metrics)
  - [Fairlearn algorithms](#fairlearn-algorithms)
- [Install Fairlearn](#install-fairlearn)
- [Usage](#usage)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [Issues](#issues)

## Current release

- The current stable release is available at
  [Fairlearn v0.7.0](https://github.com/fairlearn/fairlearn/tree/release/v0.7.0).

- Our current version may differ substantially from earlier versions.
  Users of earlier versions should visit our
  [migration guide](https://fairlearn.org/main/user_guide/mitigation.html).

## What we mean by _fairness_

An AI system can behave unfairly for a variety of reasons. In Fairlearn, we define whether an AI system is behaving unfairly in terms of its impact on people &ndash; i.e., in terms of harms. We focus on two kinds of harms:

- _Allocation harms._ These harms can occur when AI systems extend or withhold opportunities, resources, or information. Some of the key applications are in hiring, school admissions, and lending.

- _Quality-of-service harms._ Quality of service refers to whether a system works as well for one person as it does for another, even if no opportunities, resources, or information are extended or withheld.

We follow the approach known as **group fairness**, which asks: _Which groups of individuals are at risk for experiencing harms?_ The relevant groups need to be specified by the data scientist and are application specific.

Group fairness is formalized by a set of constraints, which require that some aspect (or aspects) of the AI system's behavior be comparable across the groups. The Fairlearn package enables assessment and mitigation of unfairness under several common definitions.
To learn more about our definitions of fairness, please visit our
[user guide on Fairness of AI Systems](https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html#fairness-of-ai-systems).

>_Note_:
> Fairness is fundamentally a sociotechnical challenge. Many aspects of fairness, such as justice and due process, are not captured by quantitative fairness metrics. Furthermore, there are many quantitative fairness metrics which cannot all be satisfied simultaneously. Our goal is to enable humans to assess different mitigation strategies and then make trade-offs appropriate to their scenario.

## Overview of Fairlearn

The Fairlearn Python package has two components:

- _Metrics_ for assessing which groups are negatively impacted by a model, and for comparing multiple models in terms of various fairness and accuracy metrics.

- _Algorithms_ for mitigating unfairness in a variety of AI tasks and along a variety of fairness definitions.

### Fairlearn metrics

Check out our in-depth
[guide on the Fairlearn metrics](https://fairlearn.org/main/user_guide/assessment.html).

### Fairlearn algorithms

For an overview of our algorithms please refer to our [website](https://fairlearn.org/main/user_guide/mitigation.html).

## Install Fairlearn

For instructions on how to install Fairlearn check out our [Quickstart guide](https://fairlearn.org/main/quickstart.html).

## Usage

For common usage refer to the [Jupyter notebooks](./notebooks) and our
[user guide](https://fairlearn.org/main/user_guide/index.html).
Please note that our APIs are subject to change, so notebooks downloaded
from `main` may not be compatible with Fairlearn installed with `pip`.
In this case, please navigate the tags in the repository
(e.g. [v0.4.5](https://github.com/fairlearn/fairlearn/tree/v0.4.5))
to locate the appropriate version of the notebook.

## Contributing

To contribute please check our
[contributor guide](https://fairlearn.org/main/contributor_guide/index.html).

## Maintainers

The Fairlearn project is maintained by:

- **@adrinjalali**
- **@hildeweerts**
- **@MiroDudik**
- **@mmadaio**
- **@riedgar-ms**
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

To report security issues please send an email to
`fairlearn-internal@python.org`.
