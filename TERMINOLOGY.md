# Terminology

## Estimators and predictors
Fairlearn largely follows the [terminology established by scikit-learn](https://scikit-learn.org/stable/developers/contributing.html#different-objects), specifically:
- Estimator: implements a `fit` method
- Predictor: implements a `predict` method.
Unlike [scikit-learn estimators](https://scikit-learn.org/stable/glossary.html#term-estimator) our estimators in fairlearn may not be deterministic. Lots of disparity mitigation techniques require randomized output to meet their objective in expectation. We do provide unrandomized versions of every method as well, but they may not be able to uphold the same theoretical guarantees.

## Randomized predictors

Rather than providing deterministic output many predictors produced by disparity mitigation techniques in the fairness literature are randomized. As a consequence, it is possible to get different output from the predictor's `predict` method when providing samples with identical features.

## Group Fairness

### Protected or sensitive attribute, group, or data

In the scientific literature concerning fairness one usually selects a single feature or multiple features of a dataset as a protected or sensitive set of attributes. The objective is to optimize a chosen disparity metric (for a definition see one of the later sections) based on these attributes. In some scenarios these attributes may also be held out from the actual training data (note that holding out data is not sufficient for fairness due to the myriad of correlations in real datasets). Overall, every combination of the attributes defines what we refer to as a group.
There is no definitive standard yet, but for this repository, we shall use the variable name `group_data` for a matrix of these attributes. This choice provides a few advantages:

- The term ‘protected attribute’ has a specific meaning in legal contexts which does not always match the data science usage of the term
- It may signify the presence of multiple sensitive attributes (as a matrix of values), as opposed to other choices such as `sensitive_attribute`.
- It does not falsely imply ordering between `X`, `y`, and `group_data`, as opposed to, for example, `Z`.
- It does not falsely imply that it is always part of the training features, as opposed to variable names that subscript `X`.

### Parity/Disparity vs. Fairness/Unfairness vs. Bias

There are a number of terms in the literature used to describe concepts related to disparity and fairness. As noted above, some can be particularly fraught since in legal circles they have precise meanings. We shall use the following definitions:

- *Disparity* is something that we can _measure_ and potentially _mitigate_. For example, if we see that a credit scoring model is favoring one group over another.
- *Fairness* includes the societal context in ways which may not be mathematically expressible. For example, if a finite amount of money is to be loaned, is it better for society to make a few large loans with minimal disparity, or to accept larger disparity and make many smaller loans?
- *Bias* is anything which can affect the model. Examples include statistical biases in the input data and subconscious cognitive biases in the data scientist.

### Parity Criteria/Constraints and Disparity Metrics

When trying to mitigate disparity we usually have a parity criterion in mind. For certain methods (that we refer to as reductions) these are expressed as constraints, and we optimize the objective subject to these constraints. Below are some examples of parity criteria we use in this repository:

- Classification:
    - Demographic Parity (DP): A classifier h satisfies demographic parity under a distribution over (X, A, Y) if its prediction h(X) is statistically independent of the group attribute A — that is, if P[h(X) = y’ | A = a] = P[h(X) = y’] for all a, y’. [[Agarwal et al.]](https://arxiv.org/pdf/1803.02453.pdf)
    - Equalized Odds (EO): A classifier h satisfies equalized odds under a distribution over (X, A, Y) if its prediction h(X) is conditionally independent of the group attribute A given the label Y —that is, if P[h(X) = y’ | A = a, Y = y] = P[h(X) = y’ | Y = y] for all a, y, and y’. [[Agarwal et al.]](https://arxiv.org/pdf/1803.02453.pdf)
    - Equal Opportunity is a relaxed version of Equalized Odds that only considers positive labels, i.e. Y=1, see [[Hardt et al.]]( https://ttic.uchicago.edu/~nati/Publications/HardtPriceSrebro2016.pdf)
- Regression:
    - Statistical Parity: A predictor f satisfies statistical parity under a distribution over (X, A, Y) if f(X) is independent of the group attribute A. Since 0 ≤ f(X) ≤ 1, this is equivalent to P[f(X) ≥ z | A = a] = P[f(X) ≥ z] for all a in A and z in [0,1]. [[Agarwal et al.]]( https://arxiv.org/pdf/1905.12843.pdf)
    - Bounded Group Loss: A predictor f satisfies bounded group loss at level ζ under a distribution over (X, A, Y ) if E[loss(Y, f(X)) | A = a] ≤ ζ for all a. [[Agarwal et al.]]( https://arxiv.org/pdf/1905.12843.pdf)

When assessing fairness of models we can measure disparity between groups in a variety of ways. While the criteria above are either fulfilled or not, a disparity metric provides a numerical value with which we can interpret fairness and compare models. We provide the functionality to convert common metrics from scikit-learn to group metrics, i.e. metrics that are evaluated on the entire data set, but also on each group individually. Additionally, group metrics tell us the minimum and maximum metric value and for which groups these values were observed. In certain cases we might be interested in the spread between minimum and maximum metric value, or the ratio between them. All of these are provided by group metric objects. For more information refer to `fairlearn.metrics.metrics_engine.py`.

### Fairness assessment

Fairness assessment is the process of determining and evaluating the fairness of a model through a variety of metrics in multiple configurations:

- Comparison of generated models with a visualization of the fairness/accuracy tradeoff
- Fairness and accuracy regarding training labels
- Fairness with respect to predicted outcome
