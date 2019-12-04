# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module for Fairlearn Dashboard widget."""

from .fairlearn_dashboard import FairlearnDashboard

__all__ = ['FairlearnDashboard']


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'static',
        'dest': 'fairlearn-widget',
        'require': 'fairlearn-widget/extension'
    }]
