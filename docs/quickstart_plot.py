# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Produce plot of selection rates for the quickstart guide."""
from bokeh.plotting import figure, show
from fairlearn.metrics import selection_rate_group_summary
from fairlearn.datasets import fetch_adult


data = fetch_adult(as_frame=True)
X = data.data
y_true = (data.target == '>50K') * 1
sex = X['sex']

selection_rates = selection_rate_group_summary(
    y_true, y_true, sensitive_features=sex)

xs = list(selection_rates.by_group.keys())
ys = [selection_rates.by_group[s] for s in xs]

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
