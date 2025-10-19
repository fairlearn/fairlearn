# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Script to install the requirements files, optionally pinning versions."""

import argparse
import logging
import subprocess
import sys

from _utils import _LogWrapper

_REQUIREMENTS_STEMS = ["requirements", "requirements-dev"]
_MIN_REQUIREMENTS_STEM = ["requirements-min"]

_REQUIREMENTS_EXTENSION = "txt"

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _build_argument_parser():
    desc = "Install requirements using pip or uv (preferred if available)."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--min",
        help="Install fairlearn with minimum package versions",
        action="store_true",
    )
    parser.add_argument(
        "--loglevel",
        help="Set log level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser


def _detect_installer():
    """Return ['uv', 'pip'] or ['pip'] depending on what's available."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        _logger.info("Detected uv; using 'uv pip' for installation.")
        return ["uv", "pip"]
    except (FileNotFoundError, subprocess.CalledProcessError):
        _logger.info("uv not found; falling back to pip.")
        return ["pip"]


def _install_requirements_file(file_stem):
    _logger.info("Processing %s", file_stem)

    requirements_file = f"{file_stem}.{_REQUIREMENTS_EXTENSION}"
    installer = _detect_installer()

    with _LogWrapper(f"Running {' '.join(installer)} install -r {requirements_file}"):
        command_args = installer + ["install", "-r", requirements_file]
        subprocess.check_call(command_args)


def main(argv):
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    if args.loglevel:
        logging.basicConfig(level=getattr(logging, args.loglevel))

    # Choose requirements files based on --min argument
    requirements_stems = _MIN_REQUIREMENTS_STEM if args.min else _REQUIREMENTS_STEMS
    for rs in requirements_stems:
        _install_requirements_file(rs)
    _logger.info("All requirements files installed")


if __name__ == "__main__":
    main(sys.argv[1:])
