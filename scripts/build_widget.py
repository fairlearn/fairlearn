"""Fairlearn dashboard widget build scripts.

To build the widget in order to validate local changes to the visualizations
add the --use-local-changes option and run `pip install .` after completion.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys

from _utils import _ensure_cwd_is_fairlearn_root_dir, _LogWrapper


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

_widget_js_directory = os.path.join("fairlearn", "widget", "js")
_visualization_directory = os.path.join("visualization", "FairnessDashboard")
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
    parser.add_argument("--use-local-changes",
                        help="Use local visualization changes instead of latest npm.",
                        required=False,
                        default=False,
                        action="store_true")

    return parser


def main(argv):
    _ensure_cwd_is_fairlearn_root_dir()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    with _LogWrapper("yarn install of dependencies"):
        subprocess.check_call([args.yarn_path, "install"],
                              cwd=os.path.join(os.getcwd(), _widget_js_directory))

    with _LogWrapper("yarn build"):
        subprocess.check_call([args.yarn_path, "build:all"],
                              cwd=os.path.join(os.getcwd(), _widget_js_directory))

    if args.use_local_changes:
        # Build visualizations from the visualization directory, and replace
        # the dependency in fairlearn/widget/js with it.
        with _LogWrapper("yarn install for visualizations"):
            subprocess.check_call([args.yarn_path, "install"],
                                  cwd=os.path.join(os.getcwd(), _visualization_directory))

        with _LogWrapper("yarn build for visualizations"):
            subprocess.check_call([args.yarn_path, "build"],
                                  cwd=os.path.join(os.getcwd(), _visualization_directory))

        rel_extension = ["node_modules", "fairlearn-dashboard", "rel"]
        with _LogWrapper("removing existing visualizations pulled from npm"):
            shutil.rmtree(os.path.join(_widget_js_directory, *rel_extension))

        with _LogWrapper("copying built visualizations into fairlearn widget dependencies"):
            shutil.copytree(os.path.join(_visualization_directory, "rel"),
                            os.path.join(_widget_js_directory, *rel_extension))

        with _LogWrapper("yarn build with copied local changes"):
            subprocess.check_call([args.yarn_path, "build"],
                                  cwd=os.path.join(os.getcwd(), _widget_js_directory))

    else:
        with _LogWrapper("removal of extra directories"):
            shutil.rmtree(os.path.join(_widget_js_directory, "dist"))
            shutil.rmtree(os.path.join(_widget_js_directory, "lib"))
            shutil.rmtree(os.path.join(_widget_js_directory, "node_modules"))

        if args.assert_no_changes:
            with _LogWrapper("comparison between old and newly generated widget files."):
                for file_path in _widget_generated_files:
                    # git diff occasionally leaves out some of the JS files so use git status
                    diff_result = subprocess.check_output(["git", "status"])
                    if file_path in diff_result.decode('utf-8'):
                        raise Exception("File {} was unexpectedly modified.".format(file_path))


if __name__ == "__main__":
    main(sys.argv[1:])
