"""Build Fairlearn documentation

The static landing page is no longer under the control of Sphinx,
since we only want one copy of it as we enable multiple documentation
versions.

This makes the documentation build a three-stage process:
1. Do the actual sphinx build
2. Copy the static pages into the output directory
3. Make a duplicate copy of a single PNG in the output directory (a logo)
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
    desc = "Build widget for fairlearn dashboard"

    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--documentation-path",
                        help="The path to the documentation sources (the directory containing conf.py)",
                        required=True)

    parser.add_argument("--output-path",
                        help="The directory for the output files (will be created)",
                        required=True)
    return parser


def main(argv):
    _ensure_cwd_is_fairlearn_root_dir()
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    with _LogWrapper("Run Sphinx-Multiversion"):
        subprocess.check_call(["sphinx-multiversion",
                               args.documentation_path,
                               args.output_path])

    with _LogWrapper("Copy static files"):
        shutil.copytree(os.path.join(args.documentation_path, landing_page_directory),
                        args.output_path,
                        dirs_exist_ok=True)

    with _LogWrapper("Copy individual PNG"):
        shutil.copy2(os.path.join(args.documentation_path, extra_png_src_path),
                     os.path.join(args.output_path, "images"))


if __name__ == "__main__":
    main(sys.argv[1:])
