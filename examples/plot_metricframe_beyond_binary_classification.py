# Copyright (c) Fairlearn contributors.
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
# Metric functions with different return types can also
# be mixed in a single :class:`~fairlearn.metrics.MetricFrame`.
# For example:

recall = functools.partial(skm.recall_score, average='macro')

mf2 = MetricFrame(metrics={'conf_mat': conf_mat,
                           'recall': recall
                           },
                  y_true=y_true,
                  y_pred=y_pred,
                  sensitive_features=s_f)

print("Overall values")
print(mf2.overall)
print("Values by group")
print(mf2.by_group)


# %%
# Non-scalar Inputs
# =================
#
# :class:`~fairlearn.metrics.MetricFrame` does not require
# its inputs to be scalars either. To demonstrate this, we
# will use an image recognition example (kindly supplied by
# Ferdane Bekmezci, Hamid Vaezi Joze and Samira Pouyanfar).
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
# be 1; if disjoint then it will be 0. A function to do this is:

def bounding_box_iou(box_A_input, box_B_input):
    # The inputs are array-likes in the form
    # [x_0, y_0, delta_x,delta_y]
    # where the deltas are positive

    boxA = np.array(box_A_input)
    boxB = np.array(box_B_input)

    assert boxA[2] >= 0, "Bad delta_x for boxA"
    assert boxA[3] >= 0, "Bad delta y for boxA"
    assert boxB[2] >= 0, "Bad delta x for boxB"
    assert boxB[3] >= 0, "Bad delta y for boxB"

    # Convert deltas to co-ordinates
    boxA[2:4] = boxA[0:2] + boxA[2:4]
    boxB[2:4] = boxB[0:2] + boxB[2:4]

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if (xB < xA) or (yB < yA):
        return 0

    # Compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # Compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

# %%
# This is a metric for two bounding boxes, but for :class:`~fairlearn.metrics.MetricFrame`
# we need to compare two lists of bounding boxes. For the sake of
# simplicity, we will return the mean value of 'iou' for the
# two lists, but this is by no means the only choice:


def mean_iou(true_boxes, predicted_boxes):
    assert len(true_boxes) == len(predicted_boxes), "Array size mismatch"

    all_iou = [
        bounding_box_iou(true_boxes[i], predicted_boxes[i])
        for i in range(len(true_boxes))
    ]

    return np.mean(all_iou)

# %%
# We need to generate some input data, so first create a function to
# generate a single random bounding box:


def generate_bounding_box(max_coord, max_delta):
    rng = np.random.default_rng()

    corner = max_coord * rng.random(size=2)
    delta = max_delta * rng.random(size=2)

    return np.concatenate((corner, delta))

# %%
# Now use this to create sample `y_true` and `y_pred` arrays of
# bounding boxes:


def many_bounding_boxes(n_rows, max_coord, max_delta):
    return [
        generate_bounding_box(max_coord, max_delta)
        for _ in range(n_rows)
    ]


true_bounding_boxes = many_bounding_boxes(n_rows, 5, 10)
pred_bounding_boxes = many_bounding_boxes(n_rows, 5, 10)

# %%
# Finally, we can use these in a :class:`~fairlearn.metrics.MetricFrame`:

mf_bb = MetricFrame(metrics={'mean_iou': mean_iou},
                    y_true=true_bounding_boxes,
                    y_pred=pred_bounding_boxes,
                    sensitive_features=s_f)

print("Overall metric")
print(mf_bb.overall)
print("Metrics by group")
print(mf_bb.by_group)

# %%
# The individual entries in the `y_true` and `y_pred` arrays
# can be arbitrarily complex. It is the metric functions
# which give meaning to them. Similarly,
# :class:`~fairlearn.metrics.MetricFrame` does not impose
# restrictions on the return type. One can envisage an image
# recognition task where there are multiple faces in each
# picture, and the image recognition algorithm produces
# multiple bounding boxes (not necessarily in a 1-to-1
# mapping either). The output of such a scenario might
# well be a matrix of some description.

# %%
# Conclusion
# ==========
#
# This notebook has given some taste of the flexibility
# of :class:`~fairlearn.metrics.MetricFrame` when it comes
# to inputs, outputs and metric functions.
# The input arrays can have elements of arbitrary types,
# and the return values from the metric functions can also
# be of any type (although methods such as
# :meth:`~fairlearn.metrics.MetricFrame.group_min` may not
# work).
