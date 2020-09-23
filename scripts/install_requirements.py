# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Script to install the requirements files, optionally pinning versions."""

import argparse
import logging
import os
import subprocess
import sys

from _utils import _LogWrapper

_REQUIREMENTS_STEMS = [
    "requirements",
    "requirements-customplots",
    "requirements-dev"
]

_INSERTION_FIXED = "-fixed"

_REQUIREMENTS_EXTENSION = "txt"


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _build_argument_parser():
    desc = "Install requirements using pip"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--pinned",
                        help="Whether to convert '>=' to '==' in files",
                        type=lambda x: (str(x).lower() == 'true'),  # type=bool doesn't work
                        required=True)
    parser.add_argument("--loglevel",
                        help="Set log level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    return parser


def _process_line(src_line):
    return src_line.replace('>=', '==')


def _pin_requirements(src_file, dst_file):
    with _LogWrapper("Pinning {0} into {1}".format(src_file, dst_file)):

        _logger.debug("Reading file %s", src_file)
        text_lines = []
        with open(src_file, 'r') as f:
            text_lines = f.readlines()

        result_lines = [_process_line(current_line) for current_line in text_lines]

        _logger.debug("Writing file %s", dst_file)
        with open(dst_file, 'w') as f:
            f.writelines(result_lines)


def _install_requirements_file(file_stem, fix_requirements):
    _logger.info("Processing %s", file_stem)

    if fix_requirements:
        source_file = "{0}.{1}".format(file_stem, _REQUIREMENTS_EXTENSION)
        requirements_file = "{0}{1}.{2}".format(file_stem,
                                                _INSERTION_FIXED,
                                                _REQUIREMENTS_EXTENSION)
        _pin_requirements(source_file, requirements_file)
    else:
        requirements_file = "{0}.{1}".format(file_stem, _REQUIREMENTS_EXTENSION)

    with _LogWrapper("Running pip on {0}".format(requirements_file)):
        command_args = ["pip", "install", "-r", requirements_file]
        subprocess.check_call(command_args)

    if fix_requirements:
        _logger.info("Removing temporary file %s", requirements_file)
        os.remove(requirements_file)


def main(argv):
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if args.loglevel:
        logging.basicConfig(level=getattr(logging, args.loglevel))

    _logger.info("Pinned set to: %s", args.pinned)

    for rs in _REQUIREMENTS_STEMS:
        _install_requirements_file(rs, args.pinned)
    _logger.info("All requirements files installed")


if __name__ == "__main__":
    main(sys.argv[1:])
