[![Build Status](https://dev.azure.com/responsibleai/fairlearn/_apis/build/status/Nightly?branchName=master)](https://dev.azure.com/responsibleai/fairlearn/_build/latest?definitionId=23&branchName=master) ![MIT license](https://img.shields.io/badge/License-MIT-blue.svg) ![PyPI](https://img.shields.io/pypi/v/fairlearn?color=blue)



Fairlearn
=============================================================
Fairlearn is a practical machine learning fairness toolkit that makes it easier for anyone involved in the development of an Artificial Intelligence (AI) systems to monitor and understand the impact of unfairness in their machine learning lifecycle, and use state-of-the-art techniques to mitigate the observed unfairness. Fairlearn repository contains an SDK and Jupyter notebooks with examples to showcase its use.

- [Context](#context)
- [Overview of Fairlearn](#intro)
- [Fairness Assessment Dashboard](#getting-started)
- [Unfairness Mitigation Techniques](#models)
- [Install Fairlearn](#install)
- [Contributing](#contributing)
- [Code of Conduct](#code)
- [Build Status](#build-status)
- [Additional References](#refs)


# <a name="context"></a>
# Context
We are at an age where many processes and applications have become or are becoming automated by Artificial Intelligence (AI) and machine learning systems. AI is the product of human processes and decisions used to create it, the data used to train it, and the environment used to test it.  The AI system can exhibit different, sometimes harmful, behaviors as a result of this process and systematically discriminate against people based on certain attributes protected by law. Therefore, regulated industries, or any other industry with legal, moral and ethical responsibilities, need tools to ensure the fairness of models and the predictions they produce. 
 
# <a name="intro"></a>
# Overview of Fairlearn

Fairlearn is an open source unfairness assessment and mitigation toolkit that provides business stakeholders, executives, developers, and data scientists with insights into the unfairness of their model predictions, and techniques to mitigate that. In particular, Fairlearn assists in 

1) Raising awareness of inherent unfairness in model predictions among those involved in developing AI applications by demonstrating harms visited on vulnerable groups

2) Providing state-of-the-art techniques to mitigate the observed unfairness


At the assessment phase, Fairlearn provides a dashboard with a rich set of visualizations that uses a set of disparity and accuracy metrics and user-defined protected attributes (e.g., age, gender, ethnicity) to monitors the accuracy and fairness of a model’s predictions.


At the mitigation phase, it incorporates Microsoft Research and proven third-party's state-of-the-art unfairness mitigation techniques to mitigate the unfairness observed at the previous assessment phase. Fairlearn’s mitigation stage works regression and binary classification models trained on tabular data.




# <a name="assessment"></a>
# Fairness Assessment Dashboard 

# <a name="mitigation"></a>
# Unfairness Mitigation Techniques

Fairlearn includes a Python package that implements a variety of fairness-related algorithms to mitigate bias including:

| algorithm | description | classification/regression | protected attributes | supported fairness definitions |
| --- | --- | --- | --- | --- |
| `fairlearn.` `reductions.` `ExponentiatedGradient` | Black-box approach to fair classification described in the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | categorical | DP, EO |
| `fairlearn.` `reductions.` `GridSearch` | Black-box approach described in Section 3.4 of the paper [A Reductions Approach to Fair Classification](https://arxiv.org/abs/1803.02453)| binary classification | binary | DP |
| `fairlearn.` `reductions.` `GridSearch` | Grid search for regression | regression | binary | BGL |
| `fairlearn.` `post_processing.` `ThresholdOptimizer` | Post-processing algorithm based on the paper [Equality of Opportunity in Supervised Learning](https://arxiv.org/pdf/1610.02413.pdf)| binary classification | categorical | DP, EO |


> [Note]
> DP refers to Demographic Parity, EO to Equalized Odds, and BGL to Bounded Group Loss. For more information on these and other terms we use in this repository please refer to the [terminology page](TERMINOLOGY.md). To request additional algorithms or fairness definitions, please open a new issue. 


# <a name="how-to"></a>
# Install Fairlearn
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

The function `expgrad` in the module `fairlearn.classred` implements the reduction of fair classification to weighted binary classification. Any learner that supports weighted binary classification can be provided as input for this reduction. Two common fairness definitions are provided in the module `fairlearn.moments`: demographic parity (class `DP`) and equalized odds (class `EO`). See the file `test_fairlearn.py` for example usage of `expgrad`.

# <a name="contributing"></a>

# Contributing

This project welcomes contributions and suggestions.

### Contributor License Agreement

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

To contribute please check our [Contributing guide](CONTRIBUTING.md).

# <a name="code"></a>

### Code of Conduct
This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

For common usage refer to the [Jupyter notebooks](./notebooks) and our [API guide](CONTRIBUTING.md#api)



# Maintainers

fairlearn is maintained by:

- **@MiroDudik**
- **@romanlutz**
- **@riedgar-ms**
- **@bethz**

## Releasing

If you are the current maintainer of this project:

1. Create a branch for the release: `git checkout -b release-vxx.xx`
1. Ensure that all tests return "ok": `python -m pytest -s ./test`
1. Bump the module version in `fairlearn/__init__.py`
1. Make a pull request to fairlearn/fairlearn
1. Merge fairlearn/fairlearn pull request
1. Tag and push: `git tag vxx.xx; git push --tags`

# Issues

## Regular (non-security) issues
Please submit a report through [GitHub issues](https://github.com/fairlearn/fairlearn/issues). A maintainer will respond promptly as follows:
- bug: triage as `bug` and provide estimated timeline based on severity
- feature request: triage as `feature request` and provide estimated timeline
- question or discussion: triage as `question` and respond or notify/identify a suitable expert to respond

Maintainers are supposed to link duplicate issues when possible.


## Reporting security issues

Please take a look at our guidelines for reporting [security issues](SECURITY.md).
