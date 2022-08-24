# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.


import argparse
import logging
import pathlib
import subprocess
import sys

from _utils import _LogWrapper

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def build_argument_parser():
    desc = "Build wheels for fairlearn"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--wheel-dir",
        help="Directory containing the Fairlearn wheel",
        required=True,
    )

    return parser


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    with _LogWrapper("Finding wheel file"):
        target_dir = pathlib.Path(args.wheel_dir)
        wheel_list = list(target_dir.glob("fairlearn*.whl"))
        assert len(wheel_list) == 1, f"Bad wheel_list: {wheel_list}"
        wheel_path = wheel_list[0].resolve()
        _logger.info(f"Path to wheel: {wheel_path}")

    with _LogWrapper("Installing wheel"):
        fairlearn_spec = f"fairlearn[customplots] @ {wheel_path.as_uri()}"
        subprocess.run(["pip", "install", f"{fairlearn_spec}"], check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
