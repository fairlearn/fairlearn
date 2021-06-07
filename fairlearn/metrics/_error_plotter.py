# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility class for plotting metrics with error ranges."""

from typing import Any, Callable, Dict, List, Optional, Union

from ._metric_frame import MetricFrame


class ErrorPlotter:
    """ """

    def __init__(self,
                 metric_frame: MetricFrame,
                 error_mapping: Dict[str, Dict[str, Any]]):
        """
        First level keys in `error_mapping`: Metrics (str corresponding to pandas column)
        Second level keys in `error_mapping`: "upper_bound" | "lower_bound" | "upper_error" | "lower_error" | "symmetric_error"
        Second level values in `error_mapping`: Error Metric (str corresponding to pandas column)
        
        error_mapping example:
            {
                "Recall": {
                    "upper_bound": "Recall upper bound",
                    "lower_bound": "Recall lower bound"
                },
                "Accuracy": {
                    "symmetric_error": "Accuracy error"
                }
            }

        """

        # assert verification of valid `error_mapping`

        self.metric_frame = metric_frame
        self.error_mapping = error_mapping

    def plot_with_error(self, plot_type: str, metric: str, ax = None, plot_error_labels=True, text_precision_digits=4, text_fontsize=8, text_color="black", text_ha="center", **kwargs):
        """
        Currently plots only 1 metric + its error

        """
        # assert metric is string
        assert metric in self.error_mapping.keys()
        # assert exactly one of (upper_bound, lower_bound), (upper_error, lower_error), or (symmetric_error)

        df = self.metric_frame.by_group

        # assert lower_bound < metric
        # assert upper_bound > metric

        # assert upper_error, lower_error, symmetric_error > 0

        if "upper_bound" in self.error_mapping[metric].keys() and "lower_bound" in self.error_mapping[metric].keys():
            lower_bound = self.error_mapping[metric]["lower_bound"]
            upper_bound = self.error_mapping[metric]["upper_bound"]
            
            df_lower_error = df[metric] - df[lower_bound]
            df_upper_error = df[upper_bound] - df[metric]
            df_error = [df_lower_error, df_upper_error]
        
            if plot_error_labels:
                df_lower_bound = df[lower_bound]
                df_upper_bound = df[upper_bound]
        elif "upper_error" in self.error_mapping[metric].keys() and "lower_error" in self.error_mapping[metric].keys():
            lower_error = self.error_mapping[metric]["lower_error"]
            upper_error = self.error_mapping[metric]["upper_error"]

            df_lower_error = df[lower_error]
            df_upper_error = df[upper_error]
            df_error = [df_lower_error, df_upper_error]
        
            if plot_error_labels:
                df_lower_bound = df[metric] - df_lower_error
                df_upper_bound = df[metric] + df_upper_error
        elif "symmetric_error" in self.error_mapping[metric].keys():
            symmetric_error = self.error_mapping[metric]["symmetric_error"]
            df_error = df[symmetric_error]
        
            if plot_error_labels:
                df_lower_bound = df[metric] - df_error
                df_upper_bound = df[metric] + df_error
        else:
            # TODO: Find proper error + message
            raise RuntimeError

        ax = df.plot(kind=plot_type, y=metric, yerr=df_error, ax=ax, **kwargs)

        # TODO: Check assumption of plotting items in the vertical direction
        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        if plot_error_labels:
            for i, item in enumerate(ax.patches):
                # get_width pulls left or right; get_y pushes up or down
                ax.text(i, df_lower_bound[i] - 0.05 * y_range, round(df_lower_bound[i], text_precision_digits), fontsize=text_fontsize, color=text_color, ha=text_ha)
                ax.text(i, df_upper_bound[i] + 0.01 * y_range, round(df_upper_bound[i], text_precision_digits), fontsize=text_fontsize, color=text_color, ha=text_ha)

        return ax

    def update_error_mapping():
        pass