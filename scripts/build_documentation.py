# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Build Fairlearn documentation

The static landing page is no longer under the control of Sphinx,
since we only want one copy of it as we enable multiple documentation
versions.

This makes the documentation build a three-stage process:
1. Copy the static pages into the output directory
2. Do the sphinx build
3. Make a duplicate copy of a single PNG in the output directory (a logo)

This ordering is in part because shutil.copytree() only acquired the
dirs_exist_ok argument in Python 3.8
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

landing_page_directory = "static_landing_page"
extra_png_src_path = os.path.join("_static", "images", "fairlearn_full_color.png")


def _build_argument_parser():
    desc = "Build documentation for Fairlearn"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--documentation-path",
                        help="The path to the documentation sources (conf.py directory)",
                        required=True)

    parser.add_argument("--output-path",
                        help="The directory for the output files (will be created)",
                        required=True)
    return parser


def _pip_backwards_compatibility():
    """Install extra pip packages for backwards compatibility

    This is specifically targeted at tempeh for v0.4.6.
    """
    extra_packages = ['tempeh']

    with _LogWrapper("Running pip install"):
        subprocess.check_call(["pip", "install"] + extra_packages)


def main(argv):
    _ensure_cwd_is_fairlearn_root_dir()
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    _pip_backwards_compatibility()

    with _LogWrapper("copying static files"):
        shutil.copytree(os.path.join(args.documentation_path, landing_page_directory),
                        args.output_path)

    with _LogWrapper("running Sphinx-Multiversion"):
        subprocess.check_call(["sphinx-multiversion",
                               args.documentation_path,
                               args.output_path])

    with _LogWrapper("copy of individual PNG"):
        shutil.copy2(os.path.join(args.documentation_path, extra_png_src_path),
                     os.path.join(args.output_path, "images"))


if __name__ == "__main__":
    main(sys.argv[1:])
