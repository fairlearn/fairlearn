.. _plot_metricframe:

Plotting
========

.. currentmodule:: fairlearn.metrics

Plotting grouped metrics
------------------------

The simplest way to visualize grouped metrics from the :class:`MetricFrame` is
to take advantage of the inherent plotting capabilities of
:class:`pandas.DataFrame`:

.. literalinclude:: ../../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Analyze metrics using MetricFrame
    :end-before: # Customize plots with ylim

.. figure:: ../../auto_examples/images/sphx_glr_plot_quickstart_001.png
    :target: auto_examples/plot_quickstart.html
    :align: center

It is possible to customize the plots. Here are some common examples.

Customize Plots: :code:`ylim`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The y-axis range is automatically set, which can be misleading, therefore it is
sometimes useful to set the `ylim` argument to define the yaxis range.

.. literalinclude:: ../../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Customize plots with ylim
    :end-before: # Customize plots with colormap

.. figure:: ../../auto_examples/images/sphx_glr_plot_quickstart_002.png
    :align: center


Customize Plots: :code:`colormap`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To change the color scheme, we can use the `colormap` argument. A list of colorschemes
can be found `here <https://matplotlib.org/stable/tutorials/colors/colormaps.html>`_.

.. literalinclude:: ../../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Customize plots with colormap
    :end-before: # Customize plots with kind

.. figure:: ../../auto_examples/images/sphx_glr_plot_quickstart_003.png
    :align: center

Customize Plots: :code:`kind`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are different types of charts (e.g. pie, bar, line) which can be defined by the `kind`
argument. Here is an example of a pie chart.

.. literalinclude:: ../../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Customize plots with kind
    :end-before: # Saving plots

.. figure:: ../../auto_examples/images/sphx_glr_plot_quickstart_004.png
    :align: center

There are many other customizations that can be done. More information can be found in
:meth:`pandas.DataFrame.plot`.

In order to save a plot, access the :class:`matplotlib.figure.Figure` as below and save it with your
desired filename.

.. literalinclude:: ../../auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Saving plots

Plotting with control features
------------------------------

When a :class:`MetricFrame` is created with the :code:`control_features`
argument, its :attr:`~MetricFrame.by_group` property has the control features
at the top level of its index and the sensitive features below them (see
:ref:`assessment_intersecting_groups`). Control features are for cases in which
differences between some groups are expected, and the question is whether the
metrics still differ between the sensitive groups within each control group.
In the following example, the sensitive feature is :code:`sex` and the control
feature is the age group:

.. literalinclude:: ../../auto_examples/plot_metricframe_control_features.py
    :language: python
    :start-after: # Compute metrics with control features
    :end-before: # End compute metrics with control features

Since :attr:`~MetricFrame.by_group` is still a :class:`pandas.DataFrame`, it can
be plotted as above:

.. literalinclude:: ../../auto_examples/plot_metricframe_control_features.py
    :language: python
    :start-after: # Plot all groups side by side
    :end-before: # End plot all groups side by side

.. figure:: ../../auto_examples/images/sphx_glr_plot_metricframe_control_features_001.png
    :align: center

Every combination of age group and sex is drawn as a separate bar, and nothing
in the plot shows which bars belong to the same age group. To compare the sensitive
groups within each control group, use :meth:`pandas.DataFrame.unstack` to move the
sensitive feature from the index to the columns. The bars are then clustered by
control group:

.. literalinclude:: ../../auto_examples/plot_metricframe_control_features.py
    :language: python
    :start-after: # Cluster the bars by control group
    :end-before: # End cluster the bars by control group

.. figure:: ../../auto_examples/images/sphx_glr_plot_metricframe_control_features_002.png
    :align: center

The aggregations of the :class:`MetricFrame`, such as
:meth:`~MetricFrame.difference`, are also computed within each control group, so
they return one value per control group instead of a single value. The full
:ref:`sphx_glr_auto_examples_plot_metricframe_control_features.py` example shows
these values.

A control feature only splits the data into groups. It does not account for
the reasons why the groups differ, and the data may reflect historical
discrimination, so control features should be chosen with care. Also, the more
control and sensitive groups there are, the fewer people each group has, which
makes the metrics of each group noisier (see :ref:`ci_estimation`). Including
:func:`count` as a metric shows how large each group is.

Plotting ROC curves by sensitive feature
----------------------------------------

To assess how well a binary classifier separates the positive and negative
classes for each subgroup, :func:`plot_roc_curve_by_group` draws one Receiver
Operating Characteristic (ROC) curve per group defined by the sensitive
feature(s), along with the overall curve and a chance-level baseline. Curves
that lie on top of one another indicate similar ranking performance across
groups, while diverging curves indicate that the model discriminates between
the classes better for some groups than for others.

.. literalinclude:: ../../auto_examples/plot_roc_auc.py
    :language: python
    :start-after: # Plot ROC curves by group
    :end-before: # End ROC curves by group

.. figure:: ../../auto_examples/images/sphx_glr_plot_roc_auc_001.png
    :target: auto_examples/plot_roc_auc.html
    :align: center

The function only produces the plot. To obtain the AUC scores themselves, use
:class:`MetricFrame` with :code:`sklearn.metrics.roc_auc_score`, passing the
scores as :code:`y_pred`. See the full
:ref:`sphx_glr_auto_examples_plot_roc_auc.py` example for details.
