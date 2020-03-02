"""Script for copying python version-specific requirements files to override requirements.txt.

This is for pipelines only and should not be run by developers on their machines.
"""
import argparse
import os
import platform
import shutil
import sys
from _utils import _LogWrapper


def build_argument_parser():
    desc = "Copy appropriate requirements file"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--output", help="Path to store the copied file", required=True)

    return parser


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # extract minor version, e.g. '3.5'
    py_version = ".".join(platform.python_version().split(".")[:2])
    py_version = "3.7"

    # override only if a requirements file for the specific version exists
    version_specific_requirements_file_path = "requirements-{}.txt".format(py_version)
    if os.path.exists(version_specific_requirements_file_path):
        input_file = version_specific_requirements_file_path
    else:
        input_file = 'requirements.txt'

    with _LogWrapper("Overriding {} with {}"
                     .format(args.output, input_file)):
        shutil.copyfile(input_file, args.output)

if __name__ == "__main__":
    main(sys.argv[1:])
