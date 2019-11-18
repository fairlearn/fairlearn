# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""The fairlearn package contains a number of tools for analyzing and mitigating disparity in
Machine Learning models.
"""

import os
import sys

_FAIRLEARN_RC_KEY = "FAIRLEARN_RC"


_base_version = "0.3.1"
_rc_version = ""

if _FAIRLEARN_RC_KEY in os.environ.keys():
    rc_string = os.environ[_FAIRLEARN_RC_KEY]
    try:
        rc_value = int(rc_string)
        _rc_version = "rc{0}".format(rc_value)
    except ValueError:
        msg = "Value of {0} in {1} did not parse to integer. Ignoring"
        print(msg.format(rc_string, _FAIRLEARN_RC_KEY), file=sys.stderr)


__name__ = "fairlearn"
__version__ = "{0}{1}".format(_base_version, _rc_version)


_NO_PREDICT_BEFORE_FIT = "Must call fit before attempting to make predictions"


# Setup logging infrastructure
import logging  # noqa: E402
# Only log to disk if environment variable specified
fairlearn_logs = os.environ.get('FAIRLEARN_LOGS')
if fairlearn_logs is not None:
    logging.basicConfig(filename=fairlearn_logs, level=logging.INFO)
    logging.info('Initializing logging file for fairlearn')
