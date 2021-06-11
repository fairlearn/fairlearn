"""Enables Metric Frame Plotting

The API and results of these estimators might change without any deprecation
cycle.

Importing this file dynamically sets the
:class:`~metrics._error_plotter.plot_metric_frame` as an attribute of the
`MetricFrame` module::
    >>> # explicitly require this experimental feature
    >>> from fairlearn.experimental import enable_metric_frame_plotting # noqa
    >>> # now you can import normally from metrics
    >>> from fairlearn.metrics import MetricFrame
"""

from ..metrics._plotter import (
    plot_metric_frame
)

from ..metrics import MetricFrame

# use settattr to avoid mypy errors when monkeypatching
setattr(MetricFrame, "plot_metric_frame",
        plot_metric_frame)
