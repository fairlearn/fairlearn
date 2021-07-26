# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
=========================================
MetricFrame: Beyond Binary Classification
=========================================
"""

# %%
# This notebook contains examples of using :class:`~fairlearn.metrics.MetricFrame`
# for tasks which go beyond simple binary classification.

import sklearn.metrics as skm
import functools
from fairlearn.metrics import MetricFrame

# %%
# Multiclass & Nonscalar Results
# ==============================
#
# Suppose we have a multiclass problem, with labels :math:`\in {0, 1, 2}`,
# and that we wish to generate confusion matrices for each subgroup
# identified by the sensitive feature :math:`\in { a, b, c, d}`.
# This is supported readily by
# :code:`~fairlearn.metrics.MetricFrame`, which does not require
# the result of a metric to be a scalar.
#
# First, let us generate some random input data:

import numpy as np

rng = np.random.default_rng()

n_rows = 1000
n_classes = 3
n_sensitive_features = 4

y_true = rng.integers(n_classes, size=n_rows)
y_pred = rng.integers(n_classes, size=n_rows)

temp = rng.integers(n_sensitive_features, size=n_rows)
s_f = [chr(ord('a')+x) for x in temp]

# %%
# To use :func:`~sklearn.metrics.confusion_matrix`, we
# need to prebind the `labels` argument, since it is possible
# that some of the subgroups will not contain all of
# the possible labels


conf_mat = functools.partial(skm.confusion_matrix,
                             labels=np.unique(y_true))

# %%
# With this now available, we can create our
# :class:`~fairlearn.metrics.MetricFrame`:

mf = MetricFrame(metrics={'conf_mat': conf_mat},
                 y_true=y_true,
                 y_pred=y_pred,
                 sensitive_features=s_f)

# %%
# From this, we can view the overall confusion matrix:

mf.overall

# %%
# And also the confusion matrices for each subgroup:

mf.by_group

# %%
# Obviously, the other methods such as
# :meth:`~fairlearn.metrics.MetricFrame.group_min`
# will not work, since operations such as 'less than'
# are not well defined for matrices.

# %%
# Non-scalar Inputs
# =================
#
# :class:`~fairlearn.metrics.MetricFrame` does not require
# its inputs to be scalars either. To demonstrate this, we
# will use an image recognition example (kindly supplied by
# Ferdane Bekmezci and Samira Pouyanfar).
#
# Image recognition algorithms frequently construct a bounding
# box around regions where they have found their target features.
# For example, if an algorithm detects a face in an image, it
# will place a bounding box around it. These bounding boxes
# constitute `y_pred` for :class:`~fairlearn.metrics.MetricFrame`.
# The `y_true` values then come from bounding boxes marked by
# human labellers.
#
# Bounding boxes are often compared using the 'iou' metric.
# This computes the intersection and the union of the two
# bounding boxes, and returns the ratio of their areas.
# If the bounding boxes are identical, then the metric will
# be 1; if disjoint then it will be 0.