"""Script to dynamically update the ReadMe file for a particular release

Since PyPI and GitHub have slightly different ideas about markdown, we have to update
the ReadMe file when we upload to PyPI. This script makes the necessary changes.
Most of the updates performed should be fairly robust. The one which may give trouble
is in 'update_current_version' which looks for _CURRENT_RELEASE_PATTERN in the
text in order to update both the text and the link.

The produced file assumes that a tag 'vX' (where X corresponds to the current version
of `fairlearn`) exists in the repo. Otherwise, the links won't work.
"""

import argparse
import logging
import re
import sys

logger = logging.getLogger(__name__)

_BASE_URI_FORMAT = "https://github.com/fairlearn/fairlearn/tree/v{0}"
_CURRENT_RELEASE_PATTERN = r"\[fairlearn v(\S+)\]\(https://github.com/fairlearn/fairlearn/tree/v\1\)"  # noqa: E501
_OTHER_MD_REF_PATTERN = r"\]\(\./(\w+\.md)"
_SAME_MD_REF_PATTERN = r"\]\((#.+)\)"


def build_argument_parser():
    desc = "Process ReadMe file for PyPI"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input", help="Path to the file to be processed", required=True)
    parser.add_argument("--output", help="Path to store the processed file", required=True)
    parser.add_argument("--loglevel",
                        help="Set log level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    return parser


def get_fairlearn_version():
    import fairlearn
    return fairlearn.__version__


def get_base_path(target_version):
    return _BASE_URI_FORMAT.format(target_version)


def update_current_version(line, target_version):
    logger.debug("Starting %s", sys._getframe().f_code.co_name)
    current_release_pattern = re.compile(_CURRENT_RELEASE_PATTERN)

    # Extract the current version from the line
    match = current_release_pattern.search(line)
    result = line
    if match:
        logger.info("Matched %s", match)
        # Replace with the updated version
        result = result.replace(match.groups()[0], target_version)
        logger.info("Updated string: %s", result.rstrip())
    return result


def update_other_markdown_references(line, target_version):
    logger.debug("Starting %s", sys._getframe().f_code.co_name)
    markdown_ref_pattern = re.compile(_OTHER_MD_REF_PATTERN)
    result = line
    match = markdown_ref_pattern.search(line)
    if match:
        logger.info("Matched %s", match)
        for m in match.groups():
            old_str = "./{0}".format(m)
            new_str = "{0}/{1}".format(get_base_path(target_version), m)
            result = result.replace(old_str, new_str)
        logger.info("Updated string: %s", result.rstrip())

    return result


def update_same_page_references(line, target_version):
    logger.debug("Starting %s", sys._getframe().f_code.co_name)
    same_page_ref_pattern = re.compile(_SAME_MD_REF_PATTERN)
    result = line
    match = same_page_ref_pattern.search(line)
    if match:
        logger.info("Matched %s", match)
        for m in match.groups():
            old_str = m
            new_str = "{0}{1}".format(get_base_path(target_version), m)
            result = result.replace(old_str, new_str)
        logger.info("Updated string: %s", result.rstrip())

    return result


def process_line(line, target_version):
    logger.debug("Starting %s", sys._getframe().f_code.co_name)
    result = update_current_version(line, target_version)
    result = update_other_markdown_references(result, target_version)
    result = update_same_page_references(result, target_version)

    return result


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.loglevel:
        logging.basicConfig(level=getattr(logging, args.loglevel))

    target_version = get_fairlearn_version()
    logger.info("fairlearn version: %s", target_version)

    logger.debug("Reading file %s", args.input)
    text_lines = []
    with open(args.input, 'r') as f:
        text_lines = f.readlines()

    result_lines = [process_line(l, target_version) for l in text_lines]

    logger.debug("Writing file %s", args.output)
    with open(args.output, 'w') as f:
        f.writelines(result_lines)
    logger.debug("Completed")


if __name__ == "__main__":
    main(sys.argv[1:])
