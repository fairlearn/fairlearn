The Reductions Approach to Disparity Mitigation
===============================================

The "Reductions Approach" is named because it *reduces* the problem of training a
classifier while satisfying a disparity constraint to that of training a classifier
on weighted data.
This technique was first described in [#1]_.


Fair Classification
-------------------

Consider the standard setting for binary classification.
In such a case, we wish to learn the classifer
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
For the time being, we are using the true distributions, rather than the empircal
ones.
The form we choose to express the constraints is

.. math::
    M \mathbf{\mu}(h) \le \mathbf{c}
    :label: eq_constraint_function_defn

The matrix :math:`M \in \mathbb{R}^{|\mathcal{K}| \times |\mathcal{J}|}` represents
our constraints.
We define :math:`\mathbf{\mu}(h)` as a vector of conditional moments

.. math::
    \mu_j(h) = \E
        \left[ 
            g_j(X, A, Y, h(X)) | \mathcal{E}_j
        \right ] \qquad j \in \mathcal{J}
    :label: eq_moment_definition

    g_j \, : \, \mathcal{X} \times \mathcal{A} \times \{0,1\} \times \{0,1\} \rightarrow [0,1]

and :math:`\mathcal{E}_j` is an event defined with respect to :math:`(X, A, Y)`.
Note that the function :math:`g` depends on the classifier :math:`h`, but the event
:math:`\mathcal{E}_j` does not have any such dependency.

Example Moments
^^^^^^^^^^^^^^^

We will now work through the construction of moments for some common
disparity constraints.

Demographic Parity
""""""""""""""""""

For a binary classification problem, demographic parity requires that

.. math::
    \E [ h(X)| A = a] = \E[ h(X) ]

This is a set of :math:`|\mathcal{A}|` equality constraints.
The relevant set of events :math:`\mathcal{E}_j` has one entry
:math:`\mathcal{E}_a` for each :math:`a \in \mathcal{A}`, plus
the event :math:`\mathcal{E}_{\star}` which encompasses the
entirety of the :math:`(X, A, Y)` space (since that is on the
right hand side of the definition of demographic parity given
above).
This means that :math:`\mathcal{J} = \mathcal{A} \cup \{ \star \}`.

If we set :math:`g_j(X, A, Y, h(X)) := h(x)` then, substituting
in to :eq:`eq_moment_definition` we see that
:math:`\mu_{\star}(h) = \E[ h(x) ]` and
:math:`\mu_{a}(h) = \E[ h(x) | A = a]`.
In this case, our definition of demographc parity becomes

.. math::
    \mu_{a}(h) = \mu_{\star}(h)

In order to make further progress towards the form of
equation :eq:`eq_constraint_function_defn`, we need to decide
how to measure constraint violations.
The violations can be expressed in terms of the differences between
the :math:`\mu_{a}(h)`, or in terms of the ratios between them.

First, let us express our constraints in terms of differences.
We seek to ensure that the differences in the :math:`\mu_a(h)` are
bounded by our tolerance vector :math:`\mathbf{c}`.
In this case, the demographic parity condition can be written as
a pair of inequalities:

.. math::
    \mu_{a}(h) - \mu_{\star}(h) \le c_a

    -\mu_a(h) + \mu_{\star}(h) \le c_a

where there is one pair of inequalities for each :math:`a \in \mathcal{A}`.
We have :math:`\mathcal{K} = \mathcal{A} \times \{ -, + \}`, and we can
write these constraints in the form of equation :eq:`eq_constraint_function_defn`
with:

.. math::
    M_{(a,+), a^{\prime}} & = & \mathbf{1} \{ a^{\prime} = a \} \\
    M_{(a,+), \star} & = & -1 \\
    M_{(a,-), a^{\prime}} & = & -\mathbf{1} \{ a^{\prime} = a \} \\
    M_{(a,-), \star} & = & 1

Equalized Odds
""""""""""""""

To be written.


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

Between equations :eq:`eq_fairclassify` and :eq:`eq_constraint_function_defn` the
problem we need to solve is:

.. math::
    \min_{Q \in \Delta} \hat{\mbox{err}}{(Q)}
    \quad
    \mbox{subject to}
    \quad
    M \hat{\mathbf{\mu}}(Q) \le \mathbf{\hat{c}}
    :label: eq_fairclassify_moments

Note that all of the empirical dependence of the constraint function is in
:math:`\mu`, and none is in :math:`M`.
We now form the Lagrangian:

.. math::
    L(Q, \mathbf{\lambda})
    =
    \hat{\mbox{err}}(Q)
    +
    \mathbf{\lambda}^{\mbox{T}}( M \hat{\mathbf{\mu}}(Q) - \mathbf{\hat{c}} )

The size of the vector of Lagrange multipliers, :math:`\mathbf{\lambda}`, is
set by the number of constraints, :math:`|\mathcal{K}|`.
With this Lagrangian, equation :eq:`eq_fairclassify_moments` is equivalent to:

.. math::
    \min_{Q \in \Delta}
    \max_{\mathbf{\lambda} \in \mathbb{R}_+^{|\mathcal{K}|} \; ||\mathbf{\lambda}||_1 \le B}
    L(Q, \mathbf{\lambda})
    :label: eq_saddlepoint

where the restriction to :math:`\mathbb{R}_+` comes from our choice to split
the moments into positive and negative violations of the constraint.
The restriction on the :math:`\mathcal{l}_1`-norm of :math:`\mathbf{\lambda}`
is imposed for computation and statistical reasons.
Intuitively, we are seeking to minimise our error while maximising the penalty
for violating the disparity constraint (since that penalty is controlled by
:math:`\mathbf{\lambda}` and the components of that vector are required to be
positive).
This is a saddlepoint problem.

Analysing the Saddlepoint
^^^^^^^^^^^^^^^^^^^^^^^^^

To be continued....

Solving for the Saddlepoint
---------------------------

Fairlearn contains two algorithms for solving :eq:eq_saddlepoint`.
Key to both is the ability to convert the vector of Lagrange multipliers,
:math:`\mathbf{\lambda}` into sample weights for model training.
We show how to do this in the more detailed sections below.

The simpler approach is :code:`GridSearch` which selects a collection
of :math:`\mathbf{\lambda}` vectors, and trains a model for each.
The user can then select the model which best meets their needs
(although to conform to `scikit-learn` semantics, :code:`GridSearch` will
pick one of the models to use in :code:`predict()` calls).

A fuller solution is given by the :code:`ExponentiatedGradient` algorithm.
This uses the algorithm described in [#2]_ to reach the saddlepoint.
At its core, this algorithm is a game between two players

*   The :math:`\mathbf{\lambda}`-player who seeks to maximise
    :math:`L(Q, \mathbf{\lambda})` by manipulating :math:`\mathbf{\lambda}`
    for a given model :math:`Q`
*   The :math:`Q`\-player who seeks to minimise
    :math:`L(Q, \mathbf{\lambda})` by picking the best model for a given
    :math:`\mathbf{\lambda}`

The players take turns:

#.  The :math:`\mathbf{\lambda}`-player proposes a :math:`\mathbf{\lambda}`
#.  The :math:`Q`\-player trains a model, :math:`Q`, based on the given
    :math:`\mathbf{\lambda}` and gives it back to the
    :math:`\mathbf{\lambda}`-player
#.  The :math:`\mathbf{\lambda}`-player examines how :math:`Q` violates the constraints
    and proposes a new :math:`\mathbf{\lambda}`
#.  Continue until the constraints are all satisfied

In the sections below, we will discuss the optimal strategies for each of the
two players.


Strategy for the :math:`\mathbf{\lambda}`-player
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a given :math:`Q`, the :math:`\mathbf{\lambda}`-player seeks to maximise
:math:`L(Q, \mathbf{\lambda})` by picking a valid :math:`\mathbf{\lambda}`
vector.
Recalling that this vector is constrained to have positive components and
:math:`||\mathbf{\lambda}||_1 \le B`, there are two options.
If no constraints are violated, then the best option is to set all compoments
of :math:`\mathbf{\lambda} = \mathbf{0}`.
Otherwise, the constraint on the :math:`\mathcal{l}_1`-norm means that the
best options is to put all of the weight (i.e. :math:`B`) into the element
of :math:`\mathbf{\lambda}` corresponding to the most violated constraint.

Formally, if we define:

.. math::
    \hat{\gamma}(Q) = M \hat{\mathbf{\mu}}(Q)

then the best response of the :math:`\mathbf{\lambda}`-player is given by

.. math::
    \begin{eqnarray}
    \mathbf{0} & \qquad & \mbox{if $\hat{\gamma}(Q) \le \hat{\mathbf{c}}$, }\\
    B \mathbf{e}_{k^*} & \qquad & \mbox{otherwise}
    \end{eqnarray}

where :math:`\mathbf{e}_k` is the :math:`k^{\mbox{th}}` basis vector for :math:`\mathbf{\lambda}`,
and :math:`k^* = \operatorname{argmax}_k \left [\hat{\gamma}(Q)_k - \hat{c}_k \right ]`
is the index of the most-violated constraint.

Strategy for the :math:`Q`\-player
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a given :math:`\mathbf{\lambda}`, the :math:`Q`\-player seeks to minimise
:math:`L(Q, \mathbf{\lambda})` by picking a particular form for the ensemble of
classifiers :math:`Q`.
Since :math:`L` is linear in :math:`Q`, this can always be achieved by selecting
a single classifier :math:`h \in Q`.
This then gives:

.. math::
    \begin{eqnarray}
    L(h, \mathbf{\lambda})
    & = &
    \hat{\mbox{err}}(h) + \mathbf{\lambda}^{\mbox{T}}( M \hat{\mathbf{\mu}}(h) - \mathbf{\hat{c}}) \\
    & = &
    \hat{\E} \left [ \mathbf{1} {h(X) \ne Y} \right ]
        - \mathbf{\lambda}^{\mbox{T}}\mathbf{\hat{c}}
        + \sum_{k,j} M_{k,j} \lambda_k \hat{\mu}_j (h)
    \end{eqnarray}
    :label: eq_q_player_lagrange

From equation :eq:`eq_moment_definition` we have:

.. math::
    \hat{\mu}_j(h) = \hat{\E}
        \left[ 
            g_j(X, A, Y, h(X)) | \mathcal{E}_j
        \right ]

Using the empirical event probabilities

.. math::
    p_j = \hat{\P} [ \mathcal{E}_j ]

we see that

.. math::
    \hat{\mu}_j(h) = \frac{1}{p_j}
        \hat{\E}
            \left [
                g_j(X, A, Y, h(X)) \mathbf{1} \{ (X,A,Y) \in \mathcal{E}_j \}
            \right ]

Substituting this result into equation :eq:`eq_q_player_lagrange` we find:

.. math::
    L(h, \mathbf{\lambda}) =
        - \mathbf{\lambda}^{\mbox{T}}\mathbf{\hat{c}}
        + \hat{\E} \left [ \mathbf{1} {h(X) \ne Y} \right ]
        + \sum_{k,j} \frac{M_{k,j} \lambda_k}{p_j} \hat{\E}
            \left [
                g_j(X, A, Y, h(X)) \mathbf{1} \{ (X,A,Y) \in \mathcal{E}_j \}
            \right ]
    :label: eq_q_player_lagrange_substituted

Recall that we are seeking to minimise :math:`L(h, \mathbf{\lambda})` and that
:math:`\mathbf{\lambda}` is a constant for the :math:`Q`\-player.
In this case, equation :eq:`eq_q_player_lagrange_substituted` will be
minimised by selecting :math:`h(X)` as the solution to a cost sensitive
classification problem with

.. math::
    \begin{eqnarray}
    C_i^0 & = &
        \mathbf{1} \left \{ Y_i \ne 0 \right \}
        + \sum_{k,j} \frac{M_{k,j} \lambda_k}{p_j}
                    g_j(X_i, A_i, Y_i, 0) \mathbf{1} \left \{ (X_i, A_i, Y_i) \in \mathcal{E}_j \right \} \\
    C_i^1 & = &
        \mathbf{1} \left \{ Y_i \ne 1 \right \}
        + \sum_{k,j} \frac{M_{k,j} \lambda_k}{p_j}
                    g_j(X_i, A_i, Y_i, 1) \mathbf{1} \left \{ (X_i, A_i, Y_i) \in \mathcal{E}_j \right \} \\
    \end{eqnarray}

These can be used with equation :eq:`eq_weighted_training_from_cost_sensitive` to
obtain a weighted classification problem.




.. topic:: References:

   .. [#1] Agarwal, Beygelzimer, Dudik, Langford, Wallach `"A Reductions
      Approach to Fair Classification"
      <https://arxiv.org/pdf/1803.02453.pdf>`_, ICML, 2018.

   .. [#2] Freund and Schapire, `"A Decision-Theoretic Generalization of
      On-Line Learning and an Application to Boosting" 
      <https://dl.acm.org/doi/abs/10.1006/jcss.1997.1504>`_, COLT, 1996
