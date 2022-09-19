.. _error_estimation:

Error Estimation
================

.. currentmodule:: fairlearn.metrics

It is a fact universally acknowledged that a person in possession
of a metric must be in want of an error bar.
Indeed, the omnipresence of random noise means that anything purporting
to represent the real world which does not feature an accompanying
error estimate should be regarded with suspicion.
When performing a fairness analysis this concern becomes acute,
since we divide our sample into smaller groups, increasing
the relative effects of random noise.
We are then generally interested in the difference or ratio
of function evaluations on these groups, and errors always
accumulate, even when the target values (the difference or ratio
in this case) are getting smaller.
:ref:`Intersecting groups <intersecting_groups>` make the
problem worse, since some intersections can have very low
sample counts, or even be empty.

In Fairlearn, we offer bootstrapping as a means of estimating
errors.


Bootstrapping
-------------

When analysing data, we do not (usually) have access to the
entire population; instead we have a sample.
How then should we estimate errors?
Bootstrapping is a simple approach based on
*resampling with replacement*.
The process is as follows:

#. Create a number of *bootstrap samples* by:

    #. Creating a new data set of equal size to the original
       by random sampling *with replacement*

    #. Evaluate the metric on this dataset

#. Compute the distribution function of the set
   of bootstrap samples

#. Estimate errors based on this distribution function

This is an easy and simple solution to a complex question,
so we must immediately ask ourselves "*Is this also wrong?*"
To answer this, we must first think carefully about what
we have *actually* computed.
Because we have been resampling our sample, the distribution
of bootstrap samples will be based on our sample and not
the entire population.
Hence, we should say things like
"there is a 95% likelihood that the metric *of our sample* lies between..."
and **not**
"there is a 95% likelihood that the metric lies between...."
A full analysis is beyond the scope of this user guide, but
it turns out that
*so long as our sample is representative of the population*
bootstrapping is a reasonable approach.

We then need to determine how many bootstrap samples are required.
Bootstrapping is a Monte Carlo approach, so it introduces its own
noise and reducing this will require more bootstrap samples.
In practice, it has been found that around 100 bootstrap samples
can give reasonable estimates.


Bootstrapping :code:`MetricFrame`
---------------------------------

We will now work through a short example of using :class:`MetricFrame`'s
bootstrapping capabilities.
For pedagogical purposes, we are going to work on a rigged
data set, where we know the 'true' answers in advance.
Specifically, we are going to generate data according to a binomial
distribution with :math:`p=q=0.5` and use the well known
gaussian approximations :math:`\mu = n p` and :math:`\sigma^2 = n p q`.
Our data set will have 1000 samples.
We will have two subgroups within the data set, 'A' and 'B'.
Subgroup 'A' will comprise 60% of the data, and 80% of its
values will be correct (so :math:`p_A = 0.8`).
From these, we can readily compute the corresponding values
for subgroup 'B'.

Let us start with some useful declarations:

.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> # Overall number of samples
    >>> n_samples = 1000
    >>> # Overall fraction correct
    >>> p = 0.5
    >>> # Fraction of group 'A' as proportion of whole
    >>> f_A = 0.6
    >>> # Fraction of group 'A' which are correct
    >>> p_A = 0.8

From these, we can compute a number of other
quantities.
These will be used later.

.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> # Fraction of group 'B' as proportion of whole
    >>> f_B = 1 - f_A
    >>> print(f_B)
    0.4

    >>> # Calculate numbers of each
    >>> n_A = int(f_A * n_samples)
    >>> n_B = int(f_B * n_samples)
    >>> assert n_A + n_B == n_samples
    >>> print(f"n_A={n_A}, n_B={n_B}")
    n_A=600, n_B=400

    >>> # Absolute number of group 'A' which are correct
    >>> n_A_correct = int(p_A * n_A)
    >>> print(f"n_A_correct={n_A_correct}")
    n_A_correct=480

    >>> # And for group 'B'
    >>> n_B_correct = int(n_samples * p) - n_A_correct
    >>> p_B = n_B_correct / n_B
    >>> print(f"n_B_correct={n_B_correct}")
    n_B_correct=20

    >>> # Absolute numbers which are incorrect
    >>> n_A_incorrect = n_A - n_A_correct
    >>> n_B_incorrect = n_B - n_B_correct
    >>> print(f"n_A_incorrect={n_A_incorrect}")
    n_A_incorrect=120

    >>> print(f"n_B_incorrect={n_B_incorrect}")
    n_B_incorrect=380

    >>> # And one more pair of sanity checks
    >>> assert n_A == n_A_correct + n_A_incorrect
    >>> assert n_B == n_B_correct + n_B_incorrect

For simplicity, we will use 'sum' as our metric.
Since this only requires one vector of values, but
:class:`MetricFrame` requires metrics have the
signature :code:`f(y_true, y_pred)`, we create a
wrapper.
We will ignore :code:`y_pred` in this case:


.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> import numpy as np
    >>> def bootstrap_sum(y_true, y_pred):
    ...     return np.sum(y_true)

Now, we generate the data itself.
We need a column of 0s and 1s for the 'true' values
(which we will also supply to :class:`MetricFrame`
as the 'predicted' values - our metic will ignore them),
and a column of labels 'A' and 'B' for the sensitive
feature:

.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> y_true = np.concatenate(
    ...     (np.ones(int(n_samples * p)), np.zeros(int(n_samples * (1 - p))))
    ... )
    >>> s_f = np.concatenate(
    ...     (
    ...         np.full(n_A_correct, "A"),
    ...         np.full(n_B_correct, "B"),
    ...         np.full(n_A_incorrect, "A"),
    ...         np.full(n_B_incorrect, "B"),
    ...     )
    ... )
    
    >>> # Show the sum
    >>> bootstrap_sum(y_true, y_true)
    500.0

With our data in place, we can create our :class:`MetricFrame`
object.
In order to use bootstrapping, we specify the
:code:`n_bootstrap_samples` and :code:`bootstrap_random_state`
constructor arguments.
The first determines the number of bootstrap samples, while
the second allows us to reproduce our results.

.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import MetricFrame, count
    >>> metrics = {
    ...         'bootstrap_sum': bootstrap_sum,
    ...         'count': count
    ... }
    >>> mf = MetricFrame(
    ...         metrics=metrics,
    ...         y_true=y_true,
    ...         y_pred=y_true,
    ...         sensitive_features={ 'SF0': s_f },
    ...         n_bootstrap_samples=100,
    ...         bootstrap_random_state=13489623,
    ...     )
    >>> # Show what we have:
    >>> mf.overall
        bootstrap_um     500.0
        count           1000.0
    dtype: float64
    >>> mf.by_group
            bootstrap_sum  count
        SF0
        A           480.0  600.0
        B            20.0  400.0

The underlying metrics have exactly those values expected from above.