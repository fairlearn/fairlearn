# Concepts and terminology

## Estimators and predictors

The fairlearn package largely follows the [terminology established by scikit-learn](https://scikit-learn.org/stable/developers/contributing.html#different-objects), specifically:
- _Estimators_ implement a `fit` method.
- _Predictors_ implement a `predict` method.


**Randomization.** In contrast with [scikit-learn](https://scikit-learn.org/stable/glossary.html#term-estimator), estimators in fairlearn can produce randomized predictors. Randomization of predictions is required to satisfy many definitions of fairness. Because of randomization, it is possible to get different outputs from the predictor's `predict` method on identical data. For each of our methods, we provide explicit access to the probability distribution used for randomization.

## Fairness of AI systems

AI systems can behave unfairly for a variety of reasons. Sometimes it is because of societal biases reflected in the training data and in the decisions made during the development and deployment of these systems. In other cases, AI systems behave unfairly not because of societal biases, but because of characteristics of the data (e.g., too few data points about some group of people) or characteristics of the systems themselves. Because it can be hard to distinguish between these reasons (and these reasons are not mutually exclusive and often exacerbate one another), we define whether an AI system is behaving unfairly in terms of its impact on people – i.e., in terms of harms – and not in terms of specific causes, such as societal biases, or in terms of intent, such as prejudice.

**Usage of the word _bias_.** Since we define fairness in terms of harms rather than specific causes (such as societal biases), we avoid the usage of the words _bias_ or _debiasing_ in describing the functionality of fairlearn.

### Types of harms

There are many types of harms (see, e.g., the [keynote by K. Crawford at NeurIPS 2017](https://www.youtube.com/watch?v=fMym_BKWQzk)). The fairlearn package is most applicable to two kinds of harms:

- _Allocation harms_. These harms can occur when AI systems extend or withhold opportunities, resources, or information. Some of the key applications are in hiring, school admissions, and lending.

- _Quality-of-service harms_. Quality of service refers to whether a system works as well for one person as it does for another, even if no opportunities, resources, or information are extended or withheld.

### Fairness assessment and unfairness mitigation

In fairlearn, we provide tools to assess fairness of predictors for classification and regression. We also provide tools that mitigate unfairness in classification and regression. In both assessment and mitigation scenarios, fairness is quantified using disparity metrics as we describe below.

#### Group fairness, sensitive features

There are many approaches to conceptualizing fairness. In fairlearn, we follow the approach known as group fairness, which asks: _Which groups of individuals are at risk for experiencing harms?_

The relevant groups (also called subpopulations) are defined using **sensitive features** (or sensitive attributes), which are passed to a fairlearn estimator as a vector or a matrix called `sensitive_features` (even if it is only one feature). The term suggests that the system designer should be sensitive to these features when assessing group fairness. Although these features may sometimes have privacy implications (e.g., gender or age) in other cases they may not (e.g., whether or not someone is a native speaker of a particular language). Moreover, the word sensitive does not imply that these features should not be used to make predictions – indeed, in some cases it may be better to include them.

Fairness literature also uses the term _protected attribute_ in a similar sense as sensitive feature. The term is based on anti-discrimination laws that define specific _protected classes_. Since we seek to apply group fairness in a wider range of settings, we avoid this term.

#### Parity constraints

Group fairness is typically formalized by a set of constraints on the behavior of the predictor called **parity constraints** (also called criteria). Parity constraints require that some aspect (or aspects) of the predictor behavior be comparable across the groups defined by sensitive features.

Let _X_ denote a feature vector used for predictions, _A_ be a single sensitive feature (such as age or race), and _Y_ be the true label. Parity constraints are phrased in terms of expectations with respect to the distribution over (_X,A,Y_).
For example, in fairlearn, we consider the following types of parity constraints.

_Binary classification_:

- _Demographic parity_ (also known as _statistical parity_): A classifier _h_ satisfies demographic parity under a distribution over (_X, A, Y_) if its prediction _h_(_X_) is statistically independent of the sensitive feature _A_. This is equivalent to E[_h_(_X_) | _A_=_a_] = E[_h_(_X_)] for all _a_. [[Agarwal et al.]](https://arxiv.org/pdf/1803.02453.pdf)

- _Equalized odds_: A classifier _h_ satisfies equalized odds under a distribution over (_X, A, Y_) if its prediction _h_(_X_) is conditionally independent of the sensitive feature _A_ given the label _Y_. This is equivalent to E[_h_(_X_) | _A_=_a_, _Y_=_y_] = E[_h_(_X_) | _Y_=_y_] for all _a, y_. [[Agarwal et al.]](https://arxiv.org/pdf/1803.02453.pdf)

- _Equal opportunity_: a relaxed version of equalized odds that only considers conditional expectations with respect positive labels, i.e., _Y_=1. [[Hardt et al.]]( https://ttic.uchicago.edu/~nati/Publications/HardtPriceSrebro2016.pdf)

_Regression_:

- _Demographic parity_: A predictor _f_ satisfies demographic parity under a distribution over (_X, A, Y_) if _f_(_X_) is independent of the sensitive feature _A_. This is equivalent to P[_f_(_X_) ≥ _z_ | _A_=_a_] = P[_f_(_X_) ≥ _z_] for all _a_ and _z_. [[Agarwal et al.]]( https://arxiv.org/pdf/1905.12843.pdf)

- _Bounded group loss_: A predictor _f_ satisfies bounded group loss at level _ζ_ under a distribution over (_X, A, Y_) if E[loss(_Y_, _f_(_X_)) | _A_=_a_] ≤ _ζ_ for all _a_. [[Agarwal et al.]]( https://arxiv.org/pdf/1905.12843.pdf)

Above, demographic parity seeks to mitigate allocation harms, whereas bounded group loss primarily seek to mitigate quality-of-service harms. Equalized odds and equal opportunity can be used as a diagnostic for both allocation harms as well as quality-of-service harms.

#### Disparity metrics, group metrics

Disparity metrics evaluate how far a given predictor departs from satisfying a parity constraint. They can either compare the behavior across different groups in terms of ratios or in terms of differences. For example, for binary classification:

- _Demographic parity difference_ = (max<sub>_a_</sub> E[_h_(_X_) | _A_=_a_]) - (min<sub>_a_</sub> E[_h_(_X_) | _A_=_a_]).
- _Demographic parity ratio_ = (min<sub>_a_</sub> E[_h_(_X_) | _A_=_a_]) / (max<sub>_a_</sub> E[_h_(_X_) | _A_=_a_]).

The fairlearn package provides the functionality to convert common accuracy and error metrics from `scikit-learn` to _group metrics_, i.e., metrics that are evaluated on the entire data set and also on each group individually. Additionally, group metrics yield the minimum and maximum metric value and for which groups these values were observed, as well as the difference and ratio between the maximum and the minimum values. For more information refer to the subpackage `fairlearn.metrics`.
