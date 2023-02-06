# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Script to install the requirements files, optionally pinning versions."""

import argparse
import logging
import os
import platform
import subprocess
import sys

from _utils import _LogWrapper

_REQUIREMENTS_STEMS = [
    "requirements",
    "requirements-customplots",
    "requirements-dev"]

_INSERTION_FIXED = "-fixed"

_REQUIREMENTS_EXTENSION = "txt"

# The pipelines with pinned requirements sometimes don't work when the newer
# Python versions aren't supported by the oldest allowed package version.
# The following nested dictionary maps the Python versions to the package
# names that require an override. The values are the lowest possible override
# versions that should be used in the pinned requirements files.
_REQUIREMENTS_FIXED_EXCEPTIONS = {
    "3.10": {
        "scipy": "1.7.2"
    }
}


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _build_argument_parser():
    desc = "Install requirements using pip"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--pinned",
        help="Whether to convert '>=' to '==' in files",
        type=lambda x: (str(x).lower() == "true"),  # type=bool doesn't work
        required=True,
    )
    parser.add_argument(
        "--loglevel",
        help="Set log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser


def _process_line(src_line):
    if ">=" not in src_line:
        return src_line
    processed_line = src_line.replace(">=", "==")
    python_version = platform.python_version().split(".")[:2]
    package_name = src_line.split(">=")[0]
    replacement_version = _REQUIREMENTS_FIXED_EXCEPTIONS \
        .get(python_version, {}).get(package_name, None)
    if replacement_version is not None:
        current_version = processed_line.split("==")[1]
        _logger.debug(f"Replacing {package_name} version {current_version} "
                      f"with {replacement_version}")
        return processed_line.replace(current_version, replacement_version)
    return processed_line


def _pin_requirements(src_file, dst_file):
    with _LogWrapper(f"Pinning {src_file} into {dst_file}"):

        _logger.debug(f"Reading file {src_file}")
        text_lines = []
        with open(src_file, "r") as f:
            text_lines = f.readlines()

        result_lines = [
            _process_line(current_line) for current_line in text_lines]

        _logger.debug(f"Writing file {dst_file}")
        with open(dst_file, "w") as f:
            f.writelines(result_lines)


def _install_requirements_file(file_stem, fix_requirements):
    _logger.info(f"Processing {file_stem}")

    if fix_requirements:
        source_file = f"{file_stem}.{_REQUIREMENTS_EXTENSION}"
        requirements_file = \
            f"{file_stem}{_INSERTION_FIXED}.{_REQUIREMENTS_EXTENSION}")
        _pin_requirements(source_file, requirements_file)
    else:
        requirements_file = f"{file_stem}.{_REQUIREMENTS_EXTENSION}"

    with _LogWrapper(f"Running pip on {requirements_file}"):
        command_args = ["pip", "install", "-r", requirements_file]
        subprocess.check_call(command_args)

    if fix_requirements:
        _logger.info(f"Removing temporary file {requirements_file}")
        os.remove(requirements_file)


def main(argv):
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if args.loglevel:
        logging.basicConfig(level=getattr(logging, args.loglevel))

    _logger.info(f"Pinned set to: {args.pinned}")

    for rs in _REQUIREMENTS_STEMS:
        _install_requirements_file(rs, args.pinned)
    _logger.info("All requirements files installed")


if __name__ == "__main__":
    main(sys.argv[1:])
