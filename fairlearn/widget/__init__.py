# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Package for the fairlearn Dashboard widget."""

from ._fairlearn_dashboard import FairlearnDashboard
import logging

__all__ = ['FairlearnDashboard']


logger = logging.getLogger(__file__)
logger.warn("The fairlearn.widget module will be moved into a different "
            "package called raiwidgets with the next release.")


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'fairlearn-widget',
        'require': 'fairlearn-widget/extension'
    }]
