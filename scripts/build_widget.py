# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Fairness dashboard widget build scripts.

To build the widget in order to validate local changes to the visualizations
add the --use-local-changes option and run `pip install .` after completion.
"""

import argparse
import logging
import os
import subprocess
import sys

from _utils import _ensure_cwd_is_fairlearn_root_dir, _LogWrapper


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

_widget_js_directory = os.path.join("fairlearn", "widget", "js")
_widget_generated_files = [
    'fairlearn/widget/static/index.js',
    'fairlearn/widget/static/index.js.map',
]


def build_argument_parser():
    desc = "Build widget for fairness dashboard"

    parser = argparse.ArgumentParser(description=desc)
    # example for yarn_path: 'C:\Program Files (x86)\Yarn\bin\yarn.cmd'
    parser.add_argument("--yarn-path",
                        help="The full path to the yarn executable.",
                        required=True)

    return parser


def main(argv):
    _ensure_cwd_is_fairlearn_root_dir()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    with _LogWrapper("yarn install of dependencies"):
        subprocess.check_call([args.yarn_path, "install"],
                              cwd=os.path.join(os.getcwd(), _widget_js_directory))

    with _LogWrapper("yarn build"):
        subprocess.check_call([args.yarn_path, "build"],
                              cwd=os.path.join(os.getcwd(), _widget_js_directory))


if __name__ == "__main__":
    main(sys.argv[1:])
