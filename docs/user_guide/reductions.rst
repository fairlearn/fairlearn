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
The form we choose to express the constraints is

.. math::
    M \mathbf{\mu}(h) \le \mathbf{c}
    :label: eq_moment_definition

To be continued....

Example Moments
^^^^^^^^^^^^^^^

We will now work through the construction of moments for some common
disparity constraints.

Demographic Parity
""""""""""""""""""

Need to show both difference and ratio

Equalized Odds
""""""""""""""

Need difference and ratio


Cost Sensitive and Weighted Classification
------------------------------------------

A cost-sensitive classification algorithm takes a set of training examples
:math:`\{ ( X_i, C_i^0, C_i^1 )\}_{i=1}^n` where :math:`C_i^0` and
:math:`C_i^1` are the *costs* (also known as *losses*) associated with
predicting 0 or 1 respectively for :math:`X_i`.
The result of such a classification algorithm is a classifier
:math:`h \in \mathcal{H}` which minimises:

.. math::
    \sum_{i=1}^n h(X_i)C_i^1 + (1-h(X_i))C_i^0
    :label: eq_cost_sensitive_training

A weighted classification algorithm takes a set of weighted examples
:math:`\{ ( X_i, Y_i, W_i )\}_{i=1}^n` where
:math:`Y_i \in \{0, 1\}` and :math:`W_i \ge 0`.
The result of such an algorithm is the classifier
:math:`h \in \mathcal{H}` which minimises:

.. math::
    \sum_{i=1}^n W_i \mathbf{1} \{ h(X_i) \neq Y_i \}
    :label: eq_weighted_training

These two formulations are equivalent if we set:

.. math::
    \begin{eqnarray}
    W_i & = & \left | C_i^0 - C_i^1 \right | \\
    Y_i & = & \mathbf{1} \{ C_i^0 \ge C_i^1 \}
    \end{eqnarray}
    :label: eq_weighted_training_from_cost_sensitive

To verify, suppose we set :math:`C_i^0 = 0` and
:math:`C_i^1 = 1`.
We find :math:`W_i = 1` and :math:`Y_i = 0` - 
as we would expect, since there is no cost to
predicting 0, but there is a cost to predicting 1.
Similarly, if we have :math:`C_i^0 = 1` and
:math:`C_i^1 = 0` then :math:`W_i` is unchanged but
:math:`Y_i = 1`.
Equation :eq:`eq_weighted_training_from_cost_sensitive` will
be useful as we construct the reduction.


Formulating the Reduction
-------------------------

Between equations :eq:`eq_fairclassify` and :eq:`eq_moment_definition` the
problem we need to solve is:

.. math::
    \min_{Q \in \Delta} \hat{\mbox{err}}{(Q)}
    \quad
    \mbox{subject to}
    \quad
    M \hat{\mathbf{\mu}}(Q) \le \mathbf{\hat{c}}

Note that all of the empirical dependence of the constraint function is in
:math:`\mu`, and none is in :math:`M`.

.. topic:: References:

   .. [#1] Agarwal, Beygelzimer, Dudik, Langford, Wallach `"A Reductions
      Approach to Fair Classification"
      <https://arxiv.org/pdf/1803.02453.pdf>`_, ICML, 2018.