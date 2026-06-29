# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Script to install development dependencies, optionally pinning versions."""

import argparse
import logging
import subprocess
import sys

from _utils import _LogWrapper

_DEV_REQUIREMENTS_STEM = "requirements-dev"
_MIN_REQUIREMENTS_STEM = "requirements-min"

_REQUIREMENTS_EXTENSION = "txt"

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _build_argument_parser():
    desc = (
        "Install Fairlearn's development requirements using pip. "
        "Runtime dependencies are declared in pyproject.toml and installed "
        "via `pip install .` (or `-e .`) separately."
    )

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--min",
        help=(
            "Install Fairlearn with the minimum supported runtime and development "
            "package versions pinned in requirements-min.txt"
        ),
        action="store_true",
    )
    parser.add_argument(
        "--loglevel",
        help="Set log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser


def _install_requirements_file(file_stem):
    _logger.info("Processing %s", file_stem)

    requirements_file = "{0}.{1}".format(file_stem, _REQUIREMENTS_EXTENSION)

    with _LogWrapper("Running pip on {0}".format(requirements_file)):
        command_args = ["pip", "install", "-r", requirements_file]
        subprocess.check_call(command_args)


def main(argv):
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if args.loglevel:
        logging.basicConfig(level=getattr(logging, args.loglevel))

    # Pick the appropriate requirements file. In --min mode requirements-min.txt
    # pins both runtime and development dependencies to their minimum supported
    # versions; otherwise we install the unpinned development requirements and
    # leave runtime dependency resolution to `pip install .` against pyproject.toml.
    file_stem = _MIN_REQUIREMENTS_STEM if args.min else _DEV_REQUIREMENTS_STEM
    _install_requirements_file(file_stem)
    _logger.info("Requirements file %s installed", file_stem)


if __name__ == "__main__":
    main(sys.argv[1:])
