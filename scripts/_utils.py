# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import logging
import os

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _ensure_cwd_is_fairlearn_root_dir():
    # To ensure we're in the right directory that there's a fairlearn directory inside the
    # current working directory as well as the presence of a README.rst file.
    if not os.path.exists(os.path.join(os.getcwd(), "fairlearn")) or not os.path.exists(
        os.path.join(os.getcwd(), "README.rst")
    ):
        raise Exception(
            "Please run this from the fairlearn root directory. Current directory: {}".format(
                os.getcwd()
            )
        )


class _LogWrapper:
    def __init__(self, description):
        self._description = description

    def __enter__(self):
        _logger.info("Starting %s", self._description)

    def __exit__(self, type, value, traceback):  # noqa: A002
        # raise exceptions if any occurred
        if value is not None:
            raise value
        _logger.info("Completed %s", self._description)
