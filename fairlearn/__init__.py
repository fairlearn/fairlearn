# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The fairlearn package contains a number of tools for analyzing and mitigating disparity in
Machine Learning models.
"""

import os
import sys

# Finesse the version
_FAIRLEARN_DEV_VERSION_ENV_VAR = "FAIRLEARN_DEV_VERSION"

_base_version = "0.4.0a1"
_dev_version = ""

if _FAIRLEARN_DEV_VERSION_ENV_VAR in os.environ.keys():
    dev_version_string = os.environ[_FAIRLEARN_DEV_VERSION_ENV_VAR]
    try:
        dev_version_value = int(dev_version_string)
        if dev_version_value >= 0:
            _dev_version = ".dev{0}".format(dev_version_value)
        else:
            msg = "Value of {0} was not greater than or equal to zero. Ignoring"
            print(msg.format(_FAIRLEARN_DEV_VERSION_ENV_VAR), file=sys.stderr)
    except ValueError:
        msg = "Value of {0} in {1} did not parse to integer. Ignoring"
        print(msg.format(dev_version_string, _FAIRLEARN_DEV_VERSION_ENV_VAR), file=sys.stderr)


__name__ = "fairlearn"
__version__ = "{0}{1}".format(_base_version, _dev_version)


# Common strings
_NO_PREDICT_BEFORE_FIT = "Must call fit before attempting to make predictions"


# Setup logging infrastructure
import logging  # noqa: E402
# Only log to disk if environment variable specified
fairlearn_logs = os.environ.get('FAIRLEARN_LOGS')
if fairlearn_logs is not None:
    logging.basicConfig(filename=fairlearn_logs, level=logging.INFO)
    logging.info('Initializing logging file for fairlearn')
