import argparse
import os
import re
import requirements


desc = "Checks setup.py and pip requirements files for consistency"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('directory')

args = parser.parse_args()

target_dir = os.path.abspath(args.directory)

print("Examining {0}".format(target_dir))

file_pattern = r"requirements.*\.txt"

files = [f for f in os.listdir(target_dir) if re.match(file_pattern, f)]

print(files)
