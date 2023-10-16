---
name: Bug report
about: Create a report to help us reproduce and correct the bug
title: ''
labels: 'Bug: triage'
assignees: ''

---

<!--
Before submitting a bug, please make sure the issue hasn't been already
addressed by searching through the past issues.

If your issue is a usage question, please submit it in one of these other
channels instead:
- StackOverflow with the `fairlearn` tag:
  https://stackoverflow.com/questions/tagged/fairlearn
- Discord: https://discord.gg/R22yCfgsRn
The issue tracker is used only to report bugs and feature requests. For
questions, please use either of the above platforms. Most question issues are
closed without an answer on this issue tracker. Thanks for your understanding.
-->

#### Describe the bug
<!--
A clear and concise description of what the bug is.
-->

#### Steps/Code to Reproduce
<!--
Please add a minimal example (in the form of code) that reproduces the error.
Be as succinct as possible, do not depend on external data. In short, we are
going to copy-paste your code and we expect to get the same result as you.

Example:
```python
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LinearRegression
from fairlearn.datasets import fetch_adult

data = fetch_adult()
X = pd.get_dummies(data.data)
y = (data.target == '>50K') * 1
sensitive_features = data.data['sex']
mitigator = ExponentiatedGradient(LinearRegression(), DemographicParity())
mitigator.fit(X, y, sensitive_features=sensitive_features)
```
If the code is too long, feel free to put it in a public gist and link
it in the issue: https://gist.github.com
-->

```
Sample code to reproduce the problem
```

#### Expected Results
<!-- Example: No error is thrown. Please paste or describe the expected results.-->

#### Actual Results
<!-- Please paste or specifically describe the actual output or traceback. -->

#### Screenshots
<!-- If applicable, add screenshots to help explain your problem. -->

#### Versions
<!--
Please provide the following information:
- OS: [e.g. Windows]
- Browser (if you're reporting a bug in jupyter): [e.g. Edge, Firefox, Chrome, Safari]
- Python version: [e.g. 3.9.12]
- Fairlearn version: [e.g. 0.7.0 or installed from main branch in editable mode]
- version of Python packages: please run the following snippet and paste the output:
  ```python
  import fairlearn
  fairlearn.show_versions()
  ```
-->

<!-- Thanks for contributing! -->
