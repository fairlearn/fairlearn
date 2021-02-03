# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Tools for analyzing and mitigating disparity in Machine Learning models."""

import atexit
import logging
import os
from .show_versions import show_versions  # noqa: F401

__name__ = "fairlearn"
__version__ = "0.6.0"
_base_version = __version__  # To enable the v0.4.6 docs

# Setup logging infrastructure
# Only log to disk if environment variable FAIRLEARN_LOGS specified
fairlearn_logs = os.environ.get('FAIRLEARN_LOGS')
if fairlearn_logs is not None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.makedirs(os.path.dirname(fairlearn_logs), exist_ok=True)
    handler = logging.FileHandler(fairlearn_logs, mode='w')
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info('Initializing logging file for fairlearn')

    def close_handler():  # noqa: D103
        handler.close()
        logger.removeHandler(handler)
    atexit.register(close_handler)
