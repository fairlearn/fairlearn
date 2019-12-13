import argparse
import sys


def build_argument_parser():
    desc = "Process ReadMe file for PyPI"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input", help="Path to the file to be processed", required=True)
    parser.add_argument("--output", help="Path to store the processed file", required=True)

    return parser


def read_file_by_lines(input_file_path):
    with open(input_file_path, 'r') as f:
        return f.readlines()


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    text_lines = []
    with open(args.input, 'r') as f:
        text_lines = f.readlines()

    with open(args.output, 'w') as f:
        f.writelines(text_lines)


if __name__ == "__main__":
    main(sys.argv[1:])
