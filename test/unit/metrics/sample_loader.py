# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import json
import logging
import os

_TEST_DIR = os.path.dirname(os.path.realpath(__file__))
_DATA_DIR = "sample_dashboards"

test_logger = logging.getLogger(__name__)
test_logger.setLevel(logging.INFO)


def load_sample_dashboard(file_name):
    file_path = os.path.join(_TEST_DIR, _DATA_DIR, file_name)
    with open(file_path, 'r') as fp:
        data_dict = json.load(fp)

    return data_dict
