# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The fairlearn package contains a number of tools for
analyzing and mitigating disparity in Machine Learning models.
"""

__name__ = "fairlearn"
__version__ = "0.3.0-alpha"


_NO_PREDICT_BEFORE_FIT = "Must call fit before attempting to make predictions"


# Setup logging infrastructure
import logging  # noqa: E402
import os  # noqa: E402
# Only log to disk if environment variable specified
fairlearn_logs = os.environ.get('FAIRLEARN_LOGS')
if fairlearn_logs is not None:
    logging.basicConfig(filename=fairlearn_logs, level=logging.INFO)
    logging.info('Initializing logging file for fairlearn')
