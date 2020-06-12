The Reductions Approach to Disparity Mitigation
===============================================

The "Reductions Approach" is named because it *reduces* the problem of training a
classifier while satisfying a disparity constraint to that of training a classifier
on weighted data.
This technique was first described in [#1]_.


Fair Classification
-------------------

Consider the standard setting for binary classification.
In such as case, we wish to learn the classifer
:math:`h \in \mathcal{H}` which minimises the classification
error :math:`\mbox{err}(h) = \P[ h(X) \neq Y]`.
Now, let us suppose that we can devise a constraint function
:math:`\mathbf{C}(h)` which measures the violation of our
disparity constraint.
We will then want to learn the classifier
:math:`h \in \mathcal{H}`
which satisfies

.. math::
   \min_{h \in \mathcal{H}} \mbox{err}{(h)}
   \quad
   \mbox{subject to}
   \quad
   \mathbf{C}(h) \le \mathbf{c}
   :label: eq_fairclassify_base

where :math:`\mathbf{c}` is a (vector) tolerance of the constraint
violation.
We will specify the exact form of :math:`\mathbf{C}` in a later section.

For fairness problems, we can make better tradeoffs if we make use of
*randomized classifiers*.
A randomized classifier, :math:`Q` consists of a weighted collection
of classifiers :math:`h \in \mathcal{H}`.
To make a prediction using :math:`Q`, we first select one of the
classifiers, :math:`h`, at random (according to their weights) and then
use that :math:`h` to make the prediction.
If :math:`\Delta` represents the set of all possible weighted collections
of classifiers, then we can rewrite :eq:`eq_fairclassify_base` as:

.. math::
    \min_{Q \in \Delta} \mbox{err}{(Q)}
    \quad
    \mbox{subject to}
    \quad
    \mathbf{C}(Q) \le \mathbf{c}

There is one further consideration: we do not have access to the true
distribution :math:`\{ X, A, Y \}` but only to a set of examples
:math:`\{(X_i, A_i, Y_i)\}_{i=0}^{n}`.
Consequently, we need to replace all quantities by their empirical
counterparts.
This applies not only to the error and constraint functions, but also
to the bound on constraint violations:

.. math::
    \min_{Q \in \Delta} \hat{\mbox{err}}{(Q)}
    \quad
    \mbox{subject to}
    \quad
    \mathbf{\hat{C}}(Q) \le \mathbf{\hat{c}}
    \quad
    \mbox{where}
    \quad
    \mathbf{\hat{c}}=\mathbf{c} + \mathbf{\epsilon}
    \quad
    \epsilon_k > 0
    :label: eq_fairclassify

The vector :math:`\mathbf{\epsilon}` allows for small deviations from the
formal bound on constraint violations :math:`\mathbf{c}` to reflect the
sampling noise in the data set.


Expressing the Constraint Violation with Moments
------------------------------------------------

In this section, we will discuss how to construct :math:`\mathbf{C}(h)`, which
measures the constraint violation of a classifier :math:`h`.

To be continued....

Cost Sensitive and Weighted Classification
------------------------------------------





.. topic:: References:

   .. [#1] Agarwal, Beygelzimer, Dudik, Langford, Wallach `"A Reductions
      Approach to Fair Classification"
      <https://arxiv.org/pdf/1803.02453.pdf>`_, ICML, 2018.