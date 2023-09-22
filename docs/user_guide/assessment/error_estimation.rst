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
This is because the accuracy of the bootstrapped error estimates
*depends on how well the data sample reflects the underlying population*.


Bootstrapping :code:`MetricFrame`
---------------------------------

We will now work through a short example of using :class:`MetricFrame`'s
bootstrapping capabilities.

Need to write some more here


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