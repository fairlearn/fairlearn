.. _ci_estimation:

Confidence Interval Estimation
==============================

.. currentmodule:: fairlearn.metrics

It is a fact universally acknowledged that a person in possession
of a metric must be in want of a confidence interval.
Indeed, the omnipresence of random noise means that anything purporting
to represent the real world which does not feature an accompanying
confidence interval should be regarded with suspicion.
When performing a fairness analysis this concern becomes acute,
since we divide our sample into smaller groups, increasing
the relative effects of random noise.
We are then generally interested in the difference or ratio
of function evaluations on these groups, and noise always
accumulates, even when the target values (the difference or ratio
in this case) are getting smaller.
:ref:`Intersecting groups <assessment_intersecting_groups>` make the
problem worse, since some intersections can have very low
sample counts, or even be empty.

In Fairlearn, we offer bootstrapping as a means of estimating
confidence intervals.


Bootstrapping
-------------

When analysing data, we do not (usually) have access to the
entire population; instead we have a sample.
How then should we estimate the confidence intervals associated with
the metrics we compute?
Bootstrapping is a simple approach based on
*resampling with replacement*.
The process is as follows:

#. Create a number of *bootstrap samples* by:

    #. Creating a new data set of equal size to the original
       by random sampling *with replacement*

    #. Evaluate the metric on this dataset

#. Compute the distribution function of the set
   of bootstrap samples

#. Estimate confidence intervals based on this distribution function

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
Bootstrapping is a
`Monte Carlo approach <https://en.wikipedia.org/wiki/Monte_Carlo_method>`_,
so it introduces its own noise and reducing this will require more
bootstrap samples (assuming that a poor random number generator does not
render the exercise futile).
In practice, it has been found that around 100 bootstrap samples
can give reasonable estimates.
While the number of bootstrap samples is trivial to increase,
always remember that while this may make the answers more
*precise* (by reducing the noise due to the bootstrap sampling),
it will not necessarily make them more *accurate*.
This is because the accuracy of the bootstrapped confidence interval estimates
*depends on how well the data sample reflects the underlying population*.

There is one final subtlety to remember:
when we perform bootstrapping, we will be computing a separate confidence
interval on each quantity.
There is no reason to expect that the underlying distributions for any
given pair of quantities are the same.
If the confidence intervals for that pair of quantities overlap,
we *cannot* conclude that the quantities are statistically identical.
This is particularly important in a fairness analysis where the 'good'
case is usually equality (to give a :meth:`MetricFrame.difference` of zero
or :meth:`MetricFrame.ratio` of one).
For fairness, we must confine ourselves to considering the size of the
confidence intervals, and whether they are indicating that we need to
gather more data.


Bootstrapping :code:`MetricFrame`
---------------------------------

We will now work through a short example of using :class:`MetricFrame`'s
bootstrapping capabilities.
We start by setting up a very simple and small dataset, and a couple
of metrics:

.. doctest:: bootstrap_doc_code
    :options:  +NORMALIZE_WHITESPACE

    >>> y_true = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    >>> sf_data = ['b', 'b', 'a', 'b', 'b', 'a', 'a', 'a', 'b',
    ...            'a', 'b', 'a', 'b', 'b', 'a', 'b', 'b', 'b']
    >>>
    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', 20)
    >>> pd.set_option('display.width', 80)
    >>> from fairlearn.metrics import MetricFrame
    >>> from fairlearn.metrics import count, selection_rate
    >>> # Construct a function dictionary
    >>> my_metrics = {
    ...     'sel' : selection_rate,
    ...     'count' : count
    ... }

With everything set up, we can now construct a :class:`MetricFrame` with
bootstrapping enabled.
There are three relevant arguments for the constructor:

- :code:`n_boot`
- :code:`ci_quantiles`
- :code:`random_state`

Internally, :class:`MetricFrame` will construct :code:`n_boot` bootstrap
samples (i.e. variations of the supplied dataset generated by sampling
with replacement), according to the supplied
:code:`random_state`.
Each quantity available (such as :attr:`MetricFrame.overall` or
:meth:`MetricFrame.difference`), is then evaluated for each of the
bootstrap samples.
The distribution of each is estimated via :func:`numpy.quantile`
and the quantiles specified in :code:`ci_quantiles` extracted.
Since the quantiles are estimated from a distribution, even if the
input data are integers (such as counts), then the bootstrapped
results will always be floating point numbers.
We create our :class:`MetricFrame` thus:

.. doctest:: bootstrap_doc_code
    :options:  +NORMALIZE_WHITESPACE

    >>> # Construct a MetricFrame with bootstrapping
    >>> mf = MetricFrame(
    ...     metrics=my_metrics,
    ...     y_true=y_true,
    ...     y_pred=y_pred,
    ...     sensitive_features=sf_data,
    ...     n_boot=100,
    ...     ci_quantiles=[0.159, 0.5, 0.841],
    ...     random_state=20231019
    ... )

The quantiles we have chosen (in `ci_quantiles`) correspond to the standard
deviation and median of the distribution.
The 'normal' functionality of :class:`MetricFrame` is still available.
For example:

.. doctest:: bootstrap_doc_code
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.overall
    sel       0.555556
    count    18.000000
    dtype: float64
    >>> mf.by_group
                              sel  count
    sensitive_feature_0
    a                    0.714286    7.0
    b                    0.454545   11.0

Let us look at the features bootstrapping makes available.
First, the :attr:`MetricFrame.ci_quantiles` property records
the confidence interval quantiles which we requested:

.. doctest:: bootstrap_doc_code
    :options:  +NORMALIZE_WHITESPACE

    >>> mf.ci_quantiles
    [0.159, 0.5, 0.841]

Now, we can start looking at the quantities we have computed.
These are obtained by adding :code:`_ci` to the existing
functionality.
The result is an array, indexed by :attr:`MetricFrame.ci_quantiles`
where each element is of the same type as the non-bootstrapped
function.
For example, consider :attr:`MetricFrame.overall_ci`:

.. doctest:: bootstrap_doc_code
    :options:  +NORMALIZE_WHITESPACE

    >>> _ = [print(x, '\n--') for x in mf.overall_ci]
    sel       0.444444
    count    18.000000
    dtype: float64
    --
    sel       0.555556
    count    18.000000
    dtype: float64
    --
    sel       0.666667
    count    18.000000
    dtype: float64
    --

We see that, for the overall metrics, the bootstrapped
:code:`count()` value is unchanged in each case.
This is as we would expect: each sample is constructed to have
the same number of entries as the original.
However, the :code:`selection_rate()` metric has been found
to have values of 0.444, 0.556 and 0.667 for the quantiles
specified.
These values are in line with expectations (although note that
with small numbers and 'proportion' metrics like :code:`selection_rate()`,
the median can quickly deviate from the nominal value).
Next, we can examine :attr:`MetricFrame.by_group_ci`:

.. doctest:: bootstrap_doc_code
    :options:  +NORMALIZE_WHITESPACE

    >>> _ = [print(x, '\n--') for x in mf.by_group_ci]
                              sel  count
    sensitive_feature_0
    a                    0.500000    5.0
    b                    0.333333    9.0
    --
                              sel  count
    sensitive_feature_0
    a                    0.700000    6.5
    b                    0.440972   11.5
    --
                              sel  count
    sensitive_feature_0
    a                    0.891767    9.0
    b                    0.583333   13.0
    --

We now have much more to dig into.
Firstly, the :code:`count()` metric is showing
a variation, reflecting the fact that the resampled
data are certain to have different proportions of labels
:code:`a` and :code:`b`.
Also, the sum of the :code:`count` column no longer
has to be an integer, or equal to the total number of samples.
For the median the sum is as expected, but the individual counts
are no longer integers; this is expected, since we requested
an even number of bootstrap samples.
When we inspect the :code:`sel` column, we see that the
estimates of the median, while *close* to the nominal
values (from :attr:`MetricFrame.by_group` above), are not
equal to them.
In all cases, though, the numbers reported are intuitively reasonable.

We provide methods such as :meth:`MetricFrame.group_min_ci`,
which are similar to their non-bootstrapped counterparts.
However, they have no :code:`errors` parameter.
This parameter controls what happens when a metric returns
a result for which *less-than* is not well defined (e.g a confusion
matrix).
A bootstrapped :class:`MetricFrame` will not even get this
far, since the lack of a *less-than* operator will cause
the :func:`numpy.quantile` call to fail.

Summary
-------

Bootstrapping is a powerful and simple technique, but its
limitations must be borne in mind:

* Bootstrapping **assumes the data sample is representative
  of the population**

* **Overlapping confidence intervals do not imply statistical
  equality**. This is very important in a fairness analysis
  where we are usually hoping for equality

* Increasing the number of bootstrap samples will make the
  results more precise, but not necessarily more accurate.
  The accuracy of the results depends on the degree to which
  the supplied data are representative of the population

* As a Monte-Carlo technique, it can only be as good as the
  underlying random number generator

The first of these limitations is likely to give the most
trouble in practice.