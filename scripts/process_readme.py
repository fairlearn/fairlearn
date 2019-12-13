import argparse
import re
import sys

_BASE_URI_FORMAT = "https://github.com/fairlearn/fairlearn/tree/release/{0}"
_CURRENT_RELEASE_PATTERN = r"\[fairlearn v(\d+\.\d+\.\d+)\]\(https://github.com/fairlearn/fairlearn/tree/release/\1\)"
_OTHER_MD_REF_PATTERN = r"\]\(\./(\w+\.md)"
_SAME_MD_REF_PATTERN = r"\]\((#\w+)\)"


def build_argument_parser():
    desc = "Process ReadMe file for PyPI"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--input", help="Path to the file to be processed", required=True)
    parser.add_argument("--output", help="Path to store the processed file", required=True)
    parser.add_argument("--version", help="Target version", required=True)

    return parser


def get_base_path(target_version):
    return _BASE_URI_FORMAT.format(target_version)


def update_current_version(line, target_version):
    current_release_pattern = re.compile(_CURRENT_RELEASE_PATTERN)

    # Extract the current version from the line
    match = current_release_pattern.search(line)
    if match:
        # Replace with the updated version
        return line.replace(match.groups()[0], target_version)
    return line


def update_other_markdown_references(line, target_version):
    markdown_ref_pattern = re.compile(_OTHER_MD_REF_PATTERN)
    result = line
    match = markdown_ref_pattern.search(line)
    if match:
        print(line)
        for m in match.groups():
            old_str = "./{0}".format(m)
            new_str = "{0}/{1}".format(get_base_path(target_version), m)
            result = result.replace(old_str, new_str)

    return result





def process_line(line, target_version):
    result = update_current_version(line, target_version)
    result = update_other_markdown_references(result, target_version)

    return result


def main(argv):
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    text_lines = []
    with open(args.input, 'r') as f:
        text_lines = f.readlines()

    result_lines = [process_line(l, args.version) for l in text_lines]

    with open(args.output, 'w') as f:
        f.writelines(result_lines)


if __name__ == "__main__":
    main(sys.argv[1:])
