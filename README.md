[![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Nightly?branchName=master)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=23&branchName=master) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![PyPI](https://img.shields.io/pypi/v/fairlearn?color=blue)



# fairlearn

`fairlearn` is a machine learning fairness toolkit that makes it easier for anyone involved in the development of Artificial Intelligence (AI) systems to monitor and understand the impact of unfairness in their machine learning lifecycle, and use state-of-the-art techniques to mitigate the observed unfairness. The fairlearn repository contains an SDK and Jupyter notebooks with examples to showcase its use.

- [Overview of fairlearn](#intro)
- [Target Audience](#target)
- [Fairness Assessment Dashboard](#assessment)
- [Unfairness Mitigation Techniques](#mitigation)
- [Install fairlearn](#install)
- [Contributing](#contributing)
- [Maintainers](#maintainers)
- [Issues](#issues)

 
# <a name="intro"></a>
# Overview of fairlearn

We are at an age where many processes and applications have become or are becoming automated by AI systems. AI is the product of human processes and decisions used to create it, the data used to train it, and the environment used to test it.  The AI system can exhibit different, sometimes harmful, behaviors as a result of this process and systematically discriminate against people based on certain attributes protected by law (e.g., gender, ethnicity). Therefore, regulated industries, or any other industry with legal, moral and ethical responsibilities, need tools to ensure the fairness of models and the predictions they produce. 


`fairlearn` is an open source unfairness assessment and mitigation toolkit that provides business stakeholders, executives, developers, and data scientists with insights into the unfairness of their model predictions, and techniques to mitigate that. In particular, `fairlearn` assists in 

1) raising awareness of inherent unfairness in model predictions among those involved in developing AI applications by demonstrating harms visited on vulnerable groups

2) providing state-of-the-art techniques to mitigate the observed unfairness


At the assessment phase, fairlearn provides a dashboard with a rich set of visualizations that uses a set of disparity and accuracy metrics and user-defined sensitive features (e.g., age, gender, ethnicity) to monitor the accuracy and fairness of a model’s predictions.


At the mitigation phase, it incorporates state-of-the-art unfairness mitigation techniques to mitigate the unfairness observed at the previous assessment phase. `fairlearn`’s mitigation stage supports regression and binary classification models trained on tabular data.



# <a name="target"></a>
# Target Audience



1. Machine Learning Fairness Researchers: ML researchers can easily add their unfairness mitigation techniques to this repository and use the assessment toolkit to load their original and refined model predictions to get insights into how much they have improved the disparity metrics of interest.


2. Developers/ ML Engineers/ Data Scientists: Having a variety of curated unfairness mitigations techniques in one place makes it easier for data scientists and (ML) developers to experiment with different unfairness techniques in a seamless manner. The set of rich interactive visualizations allow developers and data scientists to easily visualize the unfairness happening in their machine learning models instead of wasting time and effort on generating customized visualizations.

3. Business Executives: The set of provided visualizations are beneficial for raising awareness among those involved in developing AI applications, allow them to audit model predictions for potential unfairness, and establish a strong governance framework around the use of AI applications.

# <a name="assessment"></a>
# Fairness Assessment Dashboard 
The assessment dashboard automatically analyzes a model’s predictions, provides user with a set of insights into how the model is treating different buckets (e.g., female, male, other gender) of a sensitive feature (e.g., gender).  



These insights are calculated via a set of disparity metrics that showcase:

1. [Disparity in Accuracy]: How the model accuracy differs across different buckets of a selected sensitive feature (e.g., how accuracy of the model differs for "females" vs. "males" vs. "other gender" data points),

1. [Disparity in Predictions] How the model predictions differ across different buckets of a selected sensitive feature (e.g., how many "females" have received prediction `Approved` on their loan application in contrast to "males" and "other gender" data points?). 

The toolkit surfaces a “wizard” flow that allows you to set up accuracy and disparity metrics and then view them alongside various visualizations on a dashboard. 

After loading the visualizations via the following API call, you will see the following steps: TBC








# <a name="mitigation"></a>
# Unfairness Mitigation Techniques





`fairlearn` is a Python package that implements a variety of fairness-related algorithms to mitigate unfairness including:

## Preprocessing and In-processing

`fairlearn` shows that given access to a learning oracle for a class H, there is an efficient algorithm to find the lowest-error distribution over classifiers in H subject to equalizing false positive rates across polynomially many subgroups.

# Fairness in machine learning - Mitigation algorithms

A Python package that implements a variety of algorithms that mitigate unfairness in supervised machine learning.

## Current Release

* The current stable release is available at [fairLearn v0.2.0](https://github.com/fairlearn/fairlearn/tree/v0.2.0).

* Our latest work differs substantively from version 0.2.0, please visit the repo for further information. In particular, look at [Existing users: How to onboard to fairlearn v0.3+](#existing).

#  Algorithms

A variety of fairness-related algorithms to mitigate unfairness are included:

| algorithm | description | classification/regression | sensitive features | supported fairness definitions |
| --- | --- | --- | --- | --- |
| `fairlearn.` `reductions.` `ExponentiatedGradient` | Black-box approach to fair classification described in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | categorical | DP, EO |
| `fairlearn.` `reductions.` `GridSearch` | Black-box approach described in Section 3.4 of the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | binary | DP |
| `fairlearn.` `reductions.` `GridSearch` | Grid search for regression | regression | binary | BGL |




## Postprocessing: 
| algorithm | description | classification/regression | sensitive features | supported fairness definitions |
| --- | --- | --- | --- | --- |
| `fairlearn.` `postprocessing.` `ThresholdOptimizer` | Postprocessing algorithm based on the paper [Equality of Opportunity in Supervised Learning](https://arxiv.org/pdf/1610.02413.pdf). This technique takes as input any classifier’s prediction and the sensitive feature, and derives a monotone transformation of the classifier’s prediction to enforce the specified parity constraints. | binary classification | categorical | DP, EO |

> [Note]
> DP refers to Demographic Parity, EO to Equalized Odds, and BGL to Bounded Group Loss. For more information on these and other terms we use in this repository please refer to the [terminology page](TERMINOLOGY.md). To request additional algorithms or fairness definitions, please open a new issue. 


# <a name="install"></a>
# Install fairlearn
## Existing Users:

Use the following guide to onboard to fairlearn v0.3+:

<details>
<summary>
<strong>
<em>
Onboarding guide
</em>
</strong>
</summary>

As of version 0.2 fairlearn contained only the exponentiated gradient method. The fairlearn repository now has a more comprehensive scope and aims to incorporate other methods as specified above. The same exponentiated gradient technique is now located under `fairlearn.reductions.ExponentiatedGradient` as a class. While in the past one could have run

```python
import numpy as np
from fairlearn.classred import expgrad
from fairlearn.moments import DP

estimator = LogisticRegression()  # or any other estimator
exponentiated_gradient_result = expgrad(X, group_data, y, estimator, constraints=DP())
positive_probabilities = exponentiated_gradient_result.best_classifier(X)
randomized_predictions = (positive_probabilities >= np.random.rand(len(positive_probabilities))) * 1
```

the equivalent operation is now

```python
from fairlearn.reductions import ExponentiatedGradient
from fairlearn.reductions.moments import DemographicParity

estimator = LogisticRegression()  # or any other estimator
exponentiated_gradient = ExponentiatedGradient(estimator, constraints=DemographicParity())
exponentiated_gradient.fit(X, y, group_data)
randomized_predictions = exponentiated_gradient.predict(X)
```

Please open a new issue if you encounter any problems.

</details>


## New Users
The package can be installed via

```python
pip install fairlearn
```

or you can clone the repository locally via

```python
git clone git@github.com:fairlearn/fairlearn.git
```

To verify that it works run

```python
pip install -r requirements.txt
python -m pytest -s ./test
```

# Usage

For common usage refer to the [Jupyter notebooks](./notebooks) and our [API guide](CONTRIBUTING.md#api)

# <a name="contributing"></a>
# Contributing

To contribute please check our [Contributing guide](CONTRIBUTING.md).

# <a name="maintainers"></a>
# Maintainers

fairlearn is maintained by:

- **@MiroDudik**
- **@riedgar-ms**
- **@rihorn2**
- **@romanlutz**
- **@bethz**


# <a name="issues"></a>
# Issues

## Regular (non-security) issues
Please submit a report through [GitHub issues](https://github.com/fairlearn/fairlearn/issues). A maintainer will respond promptly as follows:
- bug: triage as `bug` and provide estimated timeline based on severity
- feature request: triage as `feature request` and provide estimated timeline
- question or discussion: triage as `question` and respond or notify/identify a suitable expert to respond

Maintainers are supposed to link duplicate issues when possible.


## Reporting security issues

Please take a look at our guidelines for reporting [security issues](SECURITY.md).
