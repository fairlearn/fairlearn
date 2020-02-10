[![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Nightly?branchName=master)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=23&branchName=master) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![PyPI](https://img.shields.io/pypi/v/fairlearn?color=blue)

# fairlearn

The fairlearn project seeks to enable anyone involved in the development of artificial intelligence (AI) systems to assess their system's fairness and mitigate the observed unfairness. The fairlearn repository contains a Python package and Jupyter notebooks with the examples of usage.

- [Current release](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#current-release)
- [What we mean by _fairness_](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#what-we-mean-by-fairness)
- [Overview of fairlearn](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#overview-of-fairlearn)
- [Install fairlearn](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#install-fairlearn)
- [Usage](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#usage)
- [Contributing](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#contributing)
- [Maintainers](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#maintainers)
- [Issues](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#issues)

## Current release

- The current stable release is available at [fairlearn v0.4.0](https://github.com/fairlearn/fairlearn/tree/release/0.4.0).

- Our current version differs substantially from version 0.2 or earlier. Users of these older versions should visit our [onboarding guide](https://github.com/fairlearn/fairlearn/tree/release/0.4.0#onboarding-guide).

## What we mean by _fairness_

An AI system can behave unfairly for a variety of reasons. In fairlearn, we define whether an AI system is behaving unfairly in terms of its impact on people &ndash; i.e., in terms of harms. We focus on two kinds of harms:

- _Allocation harms._ These harms can occur when AI systems extend or withhold opportunities, resources, or information. Some of the key applications are in hiring, school admissions, and lending.

- _Quality-of-service harms._ Quality of service refers to whether a system works as well for one person as it does for another, even if no opportunities, resources, or information are extended or withheld.

We follow the approach known as **group fairness**, which asks: _Which groups of individuals are at risk for experiencing harms?_ The relevant groups need to be specified by the data scientist and are application specific.

Group fairness is formalized by a set of constraints, which require that some aspect (or aspects) of the AI system's behavior be comparable across the groups. The fairlearn package enables assessment and mitigation of unfairness under several common definitions.
To learn more about our definitions of fairness, please visit our [terminology page](https://github.com/fairlearn/fairlearn/tree/release/0.4.0/TERMINOLOGY.md#fairness-of-ai-systems).

>_Note_:
> Fairness is fundamentally a sociotechnical challenge. Many aspects of fairness, such as justice and due process, are not captured by quantitative fairness metrics. Furthermore, there are many quantitative fairness metrics which cannot all be satisfied simultaneously. Our goal is to enable humans to assess different mitigation strategies and then make trade-offs appropriate to their scenario.

## Overview of fairlearn

The `fairlearn` package contains the following algorithms for mitigating unfairness in binary classification and regression:

| algorithm | description | classification/regression | sensitive features | supported fairness definitions |
| --- | --- | --- | --- | --- |
| `fairlearn.` `reductions.` `ExponentiatedGradient` | Black-box approach to fair classification described in [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | categorical | DP, EO |
| `fairlearn.` `reductions.` `GridSearch` | Black-box approach described in Section 3.4 of [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | binary | DP, EO |
| `fairlearn.` `reductions.` `GridSearch` | Black-box approach that implements a grid-search variant of the algorithm described in Section 5 of [Fair Regression: Quantitative Definitions and Reduction-based Algorithms](https://arxiv.org/abs/1905.12843) | regression | binary | BGL |
| `fairlearn.` `postprocessing.` `ThresholdOptimizer` | Postprocessing algorithm based on the paper [Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413). This technique takes as input an existing classifier and the sensitive feature, and derives a monotone transformation of the classifier's prediction to enforce the specified parity constraints. | binary classification | categorical | DP, EO |

> _Note_:
> DP refers to demographic parity, EO to equalized odds, and BGL to bounded group loss. For more information on these and other terms we use in this repository please refer to the [terminology page](https://github.com/fairlearn/fairlearn/tree/release/0.4.0/TERMINOLOGY.md). To request additional algorithms or fairness definitions, please open a [new issue](https://github.com/fairlearn/fairlearn/issues).

## Install fairlearn

The package can be installed via

```python
pip install fairlearn
```

or you can clone the repository locally via

```python
git clone git@github.com:fairlearn/fairlearn.git
```
If you clone from git and wish to use the Fairness dashboard, you will need to [install Yarn](https://yarnpkg.com/lang/en/docs/install), and then do the following:
```bash
cd fairlearn/widget/js
yarn install
yarn build:all
rm -rf dist
rm -rf lib
rm -rf node_modules
```
These commands only need to be run when you want the latest version of the dashboard (after pulling from our GitHub repository).

To verify that the cloned repository works (the pip package does not include the tests), run

```python
pip install -r requirements.txt
python -m pytest -s ./test/unit
```


<details name="onboarding-guide">
<summary>
<strong>
<em>
Onboarding guide for users of version 0.2 or earlier
</em>
</strong>
</summary>

Up to version 0.2, fairlearn contained only the exponentiated gradient method. The fairlearn repository now has a more comprehensive scope and aims to incorporate other methods as specified above. The same exponentiated gradient technique is now the class `fairlearn.reductions.ExponentiatedGradient`. While in the past exponentiated gradient was invoked via

```python
import numpy as np
from fairlearn.classred import expgrad
from fairlearn.moments import DP

estimator = LogisticRegression()  # or any other estimator
exponentiated_gradient_result = expgrad(X, sensitive_features, y, estimator, constraints=DP())
positive_probabilities = exponentiated_gradient_result.best_classifier(X)
randomized_predictions = (positive_probabilities >= np.random.rand(len(positive_probabilities))) * 1
```

the equivalent operation is now

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

estimator = LogisticRegression()  # or any other estimator
exponentiated_gradient = ExponentiatedGradient(estimator, constraints=DemographicParity())
exponentiated_gradient.fit(X, y, sensitive_features=sensitive_features)
randomized_predictions = exponentiated_gradient.predict(X)
```

Please open a [new issue](https://github.com/fairlearn/fairlearn/issues) if you encounter any problems.

</details>

## Usage

For common usage refer to the [Jupyter notebooks](https://github.com/fairlearn/fairlearn/tree/release/0.4.0/notebooks) and our [API guide](https://github.com/fairlearn/fairlearn/tree/release/0.4.0/CONTRIBUTING.md#api)

## Contributing

To contribute please check our [contributing guide](https://github.com/fairlearn/fairlearn/tree/release/0.4.0/CONTRIBUTING.md).

## Maintainers

The fairlearn project is maintained by:

- **@MiroDudik**
- **@riedgar-ms**
- **@rihorn2**
- **@romanlutz**
- **@bethz**

For a full list of contributors refer to the [authors page](https://github.com/fairlearn/fairlearn/tree/release/0.4.0/AUTHORS.md)

## Issues

### Regular (non-security) issues

Please submit a report through [GitHub issues](https://github.com/fairlearn/fairlearn/issues). A maintainer will respond promptly as follows:
- **bug**: triage as `bug` and provide estimated timeline based on severity
- **feature request**: triage as `feature request` and provide estimated timeline
- **question** or **discussion**: triage as `question` and either respond or notify/identify a suitable expert to respond

Maintainers will try to link duplicate issues when possible.

### Reporting security issues

Please take a look at our guidelines for reporting [security issues](https://github.com/fairlearn/fairlearn/tree/release/0.4.0/SECURITY.md).
