# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Module for Fairlearn Dashboard widget."""

from .fairlearnDashboard import FairlearnDashboard

__all__ = ['FairlearnDashboard']


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'fairlearn-widget',
        'require': 'fairlearn-widget/extension'
    }]
