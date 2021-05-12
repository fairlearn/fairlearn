# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Package for the Fairlearn Dashboard widget."""

from ._fairlearn_dashboard import FairlearnDashboard


__all__ = ['FairlearnDashboard']


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'fairlearn-widget',
        'require': 'fairlearn-widget/extension'
    }]
