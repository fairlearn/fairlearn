# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Defines the python side of the shared state of the Fairlearn widget."""

import ipywidgets as widgets
from traitlets import Unicode, Dict


@widgets.register
class FairlearnWidget(widgets.DOMWidget):
    """The python widget definition for the Fairlearn dashboard."""

    _view_name = Unicode('FairlearnView').tag(sync=True)
    _model_name = Unicode('FairlearnModel').tag(sync=True)
    _view_module = Unicode('fairlearn-widget').tag(sync=True)
    _model_module = Unicode('fairlearn-widget').tag(sync=True)
    _view_module_version = Unicode('^0.1.1').tag(sync=True)
    _model_module_version = Unicode('^0.1.1').tag(sync=True)
    value = Dict().tag(sync=True)
    request = Dict({}).tag(sync=True)
    response = Dict({}).tag(sync=True)
