import argparse
import sys


def build_argument_parser():
    desc = "Process ReadMe file for PyPI"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input", help="Path to the file to be processed", required=True)
    parser.add_argument("--output", help="Path to store the processed file", required=True)

    return parser


def process_line(line):
    result = line
    return result


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    text_lines = []
    with open(args.input, 'r') as f:
        text_lines = f.readlines()

    result_lines = [process_line(l) for l in text_lines]

    with open(args.output, 'w') as f:
        f.writelines(result_lines)


if __name__ == "__main__":
    main(sys.argv[1:])
