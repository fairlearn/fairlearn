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
noise and reducing this will require more bootstrap samples
(assuming that a poor random number generator does not render
the exercise futile).
In practice, it has been found that around 100 bootstrap samples
can give reasonable estimates.
While the number of bootstrap samples is trivial to increase,
always remember that while this may make the answers more
*precise* (by reducing the noise due to the bootstrap sampling),
it will not necessarily make them more *accurate*.
This is because tne accuracy of the bootstrapped error estimates
*depends on how well the data sample reflects the underlying population*.


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

Constructing the DataFrame
^^^^^^^^^^^^^^^^^^^^^^^^^^

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

Activating Bootstrapping in :code:`MetricFrame`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
    ...         'bootstrap_sum' : bootstrap_sum,
    ...         'count' : count
    ... }
    >>> mf = MetricFrame(
    ...         metrics=metrics,
    ...         y_true=y_true,
    ...         y_pred=y_true,
    ...         sensitive_features={ 'SF0' : s_f },
    ...         n_bootstrap_samples=100,
    ...         bootstrap_random_state=13489623,
    ...     )
    >>> # Show what we have:
    >>> mf.overall
    bootstrap_sum     500.0
    count            1000.0
    dtype: float64
    >>> mf.by_group
        bootstrap_sum  count
    SF0
    A           480.0  600.0
    B            20.0  400.0

The underlying metrics have the nominal values we expect from above.


Accessing the Bootstrapped Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now on to the bootstrap results themselves.
We access the bootstrapped values for the overall values via
:meth:`MetricFrame.overall_quantiles`.
This method takes a single argument, :code:`quantiles`,
which is a list of the quantiles we want extracted, each
in the range :math:`[0, 1]`.
We are going to look at the bootstrapped mean and standard
deviation, so the quantiles we want are 0.159, 0.5 and 0.841
(we happen to know that the usual approximation will hold for
our data):

.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> my_quantiles = [0.159, 0.5, 0.841]
    >>> mf.overall_quantiles(quantiles=my_quantiles)
    bootstrap_sum    [483.48..., 501.0, 513.518]
    count            [1000.0, 1000.0, 1000.0]
    dtype: object

Rather than single floats, the resultant :class:`pandas.Series`
contains arrays of results.
Each entry corresponds to one of the quantiles we specified
in the call to :meth:`MetricFrame.overall_quantiles`.
The results for 'count' are not particularly interesting, since
each resampling was always the same size as the original.
However, we can check that the results for :code:`bootstrap_sum`
are in line with our expectations.
The value for the median (which is also the mean for this
symmetric distribution) is 501; we know that theoretically
it should be 500, so this estimate is correct to two
significant figures.
This is well within expectations for 100 bootstrap samples
(applying a :math:`\sqrt{N}` rule-of-thumb).
We also know that
:math:`\sigma = \sqrt{n p q} = \sqrt{1000 \times 0.5 \times 0.5} \approx  15.8`,
so we would expect the other two quantiles to be at
:math:`500-15.8=484.2` and :math:`500+15.8=515.8`,
which are again well within expectations for 100 bootstrap samples.

Similarly, we have a :meth:`MetricFrame.by_group_quantiles`
to go with the :attr:`MetricFrame.by_group` property:

.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.by_group
         bootstrap_sum  count
    SF0
    A            480.0  600.0
    B             20.0  400.0
    >>> mf.by_group_quantiles(quantiles=my_quantiles)
                 bootstrap_sum                    count
    SF0
    A    [465.0, 480.0, 495.0]  [582.741, 600.5, 615.0]
    B       [16.0, 19.5, 23.0]  [385.0, 399.5, 417.259]

We can see that the nominal values from :attr:`MetricFrame.by_group`
are as expected, given how we constructed our dataset.
The :meth:`MetricFrame.by_group_quantiles` method has
returned a similarly shaped :class:`pandas.DataFrame`, but instead
of a scalar value in each cell, we have a list of three values,
corresponding to the specified quantiles.
In each case, the value for quantile 0.5 (the second value in each
triplet) is close to the nominal value.

Next, let us consider the 'count' metric.
This is effectively considering a different binomial distribution
on the 1000 samples, but based on :math:`f_A = 0.6` and
:math:`f_B = 0.4`.
The corresponding standard deviation is
:math:`\sqrt{1000  \times0.6 \times 0.4} = 15.5`, and we can see
that the other two quantiles calculated for the 'count' are
close to the nominal value :math:`\pm 15.5`.

*Need to figure out the estimate on the bootstrap_sum, since*
:math:`\sqrt{600 \times 0.8 \times 0.2} = 9.8`
*which is smaller than the range seen.*

Now, let us consider the various methods on :class:`MetricFrame`.
Rather than using a different method, we have a :code:`quantiles`
argument on each.
For example, the :meth:`MetricFrame.group_max` function:

.. doctest:: error_estimation_code
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.group_max()
    bootstrap_sum    480.0
    count            600.0
    dtype: object
    >>> mf.group_max(quantiles=my_quantiles)
    bootstrap_sum      [465.0, 480.0, 495.0]
    count            [582.741, 600.5, 615.0]
    dtype: object

These are as expected from the results shown above.


Summary
-------

Bootstrapping is a powerful and simple technique, but its
limitations must be borne in mind:

* As a Monte-Carlo technique, it can only be as good as the
  underlying random number generator

* Increasing the number of bootstrap samples will make the
  results more precise, but not necessarily more accurate

* Bootstrapping **assumes the data sample is representative
  of the population**

The last of these limitations is likely to give the most
trouble in practice.