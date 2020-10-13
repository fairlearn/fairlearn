# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Produce plot of selection rates for the quickstart guide."""
from bokeh.plotting import figure, show
from fairlearn.metrics import MetricsFrame, selection_rate
from fairlearn.datasets import fetch_adult


data = fetch_adult(as_frame=True)
X = data.data
y_true = (data.target == '>50K') * 1
sex = X['sex']

selection_rates = MetricsFrame(selection_rate,
                                y_true, y_true,
                                sensitive_features=sex)

xs = list(selection_rates.by_group.index)
ys = [selection_rates.by_group['selection_rate'][s] for s in xs]

p = figure(x_range=xs,
           plot_height=480,
           plot_width=640,
           title="Fraction earning over $50,0000",
           toolbar_location=None,
           tools="")

p.vbar(x=xs, top=ys, width=0.9)

p.y_range.start = 0
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None

show(p)
