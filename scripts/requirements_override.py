"""Script for copying python version-specific requirements files to override requirements.txt.

This is for pipelines only and should not be run by developers on their machines.
The script checks for a requirements file that's specific to the python minor version, e.g. 3.5,
and uses those requirements, or if no such file exists the script creates that file based on
the generic requirements.txt file. Our pipelines rely on the existence of a
requirements-{major}-{minor}.txt file.
"""
import argparse
import os
import platform
import shutil
import sys
from _utils import _LogWrapper


def build_argument_parser():
    desc = "Copy appropriate requirements file (either generic or python version specific) " \
           "to requirements-{major}-{minor}.txt"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--output", help="Path to store the copied file", required=True)

    return parser


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    # extract minor version, e.g. '3.5'
    py_version = ".".join(platform.python_version().split(".")[:2])

    # override only if a requirements file for the specific version exists
    version_specific_requirements_file_path = "requirements-{}.txt".format(py_version)
    if os.path.exists(version_specific_requirements_file_path):
        input_file = version_specific_requirements_file_path
    else:
        input_file = 'requirements.txt'

    with _LogWrapper("Overriding {} with {}"
                     .format(args.output, input_file)):
        try:
            shutil.copyfile(input_file, args.output)
        except shutil.SameFileError:
            # destination is already identical with origin
            pass


if __name__ == "__main__":
    main(sys.argv[1:])
