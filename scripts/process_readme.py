# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Script to dynamically update the ReadMe file for a particular release

Since PyPI and GitHub have slightly different ideas about markdown, we have to update
the ReadMe file when we upload to PyPI. This script makes the necessary changes.
Most of the updates performed should be fairly robust. The one which may give trouble
is in '_update_current_version' which looks for _CURRENT_RELEASE_PATTERN in the
text in order to update both the text and the link.

The produced file assumes that a tag 'vX' (where X corresponds to the current version
of `fairlearn`) exists in the repo. Otherwise, the links won't work.
"""

import argparse
import logging
import os
import re
import sys

from _utils import _LogWrapper

_logger = logging.getLogger(__name__)

_BASE_URI_FORMAT = "https://github.com/fairlearn/fairlearn/tree/v{0}"
_CURRENT_RELEASE_PATTERN = (
    r"\[fairlearn v(\S+)\]\(https://github.com/fairlearn/fairlearn/tree/v\1\)"  # noqa: E501
)
_OTHER_MD_REF_PATTERN = r"\]\(\./(\w+\.md)"
_SAME_MD_REF_PATTERN = r"\]\((#.+)\)"


def _get_base_path(target_version):
    return _BASE_URI_FORMAT.format(target_version)


def _update_current_version(line, target_version):
    _logger.info("Starting %s", sys._getframe().f_code.co_name)
    current_release_pattern = re.compile(_CURRENT_RELEASE_PATTERN)

    # Extract the current version from the line
    match = current_release_pattern.search(line)
    result = line
    if match:
        _logger.info("Matched %s", match)
        # Replace with the updated version
        result = result.replace(match.groups()[0], target_version)
        _logger.info("Updated string: %s", result.rstrip())
    return result


def _update_other_markdown_references(line, target_version):
    _logger.info("Starting %s", sys._getframe().f_code.co_name)
    markdown_ref_pattern = re.compile(_OTHER_MD_REF_PATTERN)
    result = line
    match = markdown_ref_pattern.search(line)
    if match:
        _logger.info("Matched %s", match)
        for m in match.groups():
            old_str = "./{0}".format(m)
            new_str = "{0}/{1}".format(_get_base_path(target_version), m)
            result = result.replace(old_str, new_str)
        _logger.info("Updated string: %s", result.rstrip())

    return result


def _update_same_page_references(line, target_version):
    _logger.info("Starting %s", sys._getframe().f_code.co_name)
    same_page_ref_pattern = re.compile(_SAME_MD_REF_PATTERN)
    result = line
    match = same_page_ref_pattern.search(line)
    if match:
        _logger.info("Matched %s", match)
        for m in match.groups():
            old_str = m
            new_str = "{0}{1}".format(_get_base_path(target_version), m)
            result = result.replace(old_str, new_str)
        _logger.info("Updated string: %s", result.rstrip())

    return result


def _process_line(line, target_version):
    _logger.info("Starting %s", sys._getframe().f_code.co_name)
    result = _update_current_version(line, target_version)
    result = _update_other_markdown_references(result, target_version)
    result = _update_same_page_references(result, target_version)

    return result


def process_readme(input_file_name, output_file_name):
    sys.path.append(os.getcwd())
    import fairlearn

    target_version = fairlearn.__version__
    _logger.info("fairlearn version: %s", target_version)

    text_lines = []
    with _LogWrapper("reading file {}".format(input_file_name)):
        with open(input_file_name, "r") as input_file:
            text_lines = input_file.readlines()

    result_lines = [_process_line(line, target_version) for line in text_lines]

    with _LogWrapper("writing file {}".format(output_file_name)):
        with open(output_file_name, "w") as output_file:
            output_file.writelines(result_lines)


def build_argument_parser():
    desc = "Process ReadMe file for PyPI"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--input-file-name", help="Path to the file to be processed", required=True
    )
    parser.add_argument(
        "--output-file-name", help="Path to store the processed file", required=True
    )

    return parser


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    process_readme(args.input_file_name, args.output_file_name)


if __name__ == "__main__":
    main(sys.argv[1:])
