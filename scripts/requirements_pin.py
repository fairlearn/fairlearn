"""Script for changing a regular requirements.txt into a pinned one

This searches through the specified file, replacing all occurences of
a lower bound '>=' with a pinned '=='.
"""
import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def build_argument_parser():
    desc = "Pin Requirements file by converting '>=' to '=='"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input", help="Path to the file to be processed", required=True)
    parser.add_argument("--output", help="Path to store the processed file", required=True)
    parser.add_argument("--loglevel",
                        help="Set log level",
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    return parser


def process_line(src_line):
    return src_line.replace('>=', '==')


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if args.loglevel:
        logging.basicConfig(level=getattr(logging, args.loglevel))

    logger.debug("Reading file %s", args.input)
    text_lines = []
    with open(args.input, 'r') as f:
        text_lines = f.readlines()

    result_lines = [process_line(l) for l in text_lines]

    logger.debug("Writing file %s", args.output)
    with open(args.output, 'w') as f:
        f.writelines(result_lines)
    logger.debug("Completed")


if __name__ == "__main__":
    main(sys.argv[1:])
