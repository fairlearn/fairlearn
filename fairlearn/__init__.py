# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

__name__ = "fairlearn"
__version__ = "0.3.0-alpha"

_KW_SENSITIVE_FEATURES = "sensitive_features"


# Setup logging infrastructure
import logging  # noqa: E402
import os  # noqa: E402
# Only log to disk if environment variable specified
fairlearn_logs = os.environ.get('FAIRLEARN_LOGS')
if fairlearn_logs is not None:
    logging.basicConfig(filename=fairlearn_logs, level=logging.INFO)
    logging.info('Initializing logging file for fairlearn')
