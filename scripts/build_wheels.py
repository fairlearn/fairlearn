# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import argparse
import logging
import subprocess
import sys

from _utils import _ensure_cwd_is_fairlearn_root_dir, _get_fairlearn_version, _LogWrapper
from process_readme import process_readme

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def build_argument_parser():
    desc = "Build wheels for fairlearn"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--version-filename",
        help="The file where the version will be stored.",
        required=True,
    )

    return parser


def main(argv):
    _ensure_cwd_is_fairlearn_root_dir()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    with _LogWrapper("processing README.rst"):
        process_readme("README.rst", "README.rst")

    with _LogWrapper("storing fairlearn version in {}".format(args.version_filename)):
        version = _get_fairlearn_version()
        with open(args.version_filename, "w") as version_file:
            version_file.write(version)

    with _LogWrapper("creation of packages"):
        subprocess.check_call([sys.executable, "-m", "build"])


if __name__ == "__main__":
    main(sys.argv[1:])
