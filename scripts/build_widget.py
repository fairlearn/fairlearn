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
_widget_generated_files = [
    'fairlearn/widget/static/extension.js',
    'fairlearn/widget/static/extension.js.map',
    'fairlearn/widget/static/index.js',
    'fairlearn/widget/static/index.js.map',
    'jupyter-config/nbconfig/notebook.d/fairlearn-widget.json',
    'fairlearn/widget/js/fairlearn_widget/labextension/fairlearn-widget-0.1.0.tgz'
]


def build_argument_parser():
    desc = "Build widget for fairlearn dashboard"

    parser = argparse.ArgumentParser(description=desc)
    # example for yarn_path: 'C:\Program Files (x86)\Yarn\bin\yarn.cmd'
    parser.add_argument("--yarn-path",
                        help="The full path to the yarn executable.",
                        required=True)
    parser.add_argument("--assert-no-changes",
                        help="Assert that the generated files did not change.",
                        required=False,
                        default=False,
                        action='store_true')

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

    if args.assert_no_changes:
        with LogWrapper("comparison between old and newly generated widget files."):
            for file_path in _widget_generated_files:
                diff_result = subprocess.check_output(["git", "diff", os.path.abspath(file_path)])
                if diff_result == b'':
                    _logger.info("File {} has not changed.".format(file_path))
                else:
                    _logger.error("File {} has changed unexpectedly.".format(file_path))


if __name__ == "__main__":
    main(sys.argv[1:])
