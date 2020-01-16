import argparse
import logging
import os
import shutil
import subprocess
import sys

from _utils import _ensure_cwd_is_fairlearn_root_dir, LogWrapper


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

_widget_js_directory = os.path.join("fairlearn", "widget", "js")


def build_argument_parser():
    desc = "Build widget for fairlearn dashboard"

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

    with LogWrapper("yarn install of dependencies"):
        subprocess.check_call([args.yarn_path, "install"],
                              cwd=os.path.join(os.getcwd(), _widget_js_directory))

    with LogWrapper("yarn build"):
        subprocess.check_call([args.yarn_path, "build:all"],
                              cwd=os.path.join(os.getcwd(), _widget_js_directory))

    with LogWrapper("removal of extra directories"):
        shutil.rmtree(os.path.join(_widget_js_directory, "dist"))
        shutil.rmtree(os.path.join(_widget_js_directory, "lib"))
        shutil.rmtree(os.path.join(_widget_js_directory, "node_modules"))


if __name__ == "__main__":
    main(sys.argv[1:])
