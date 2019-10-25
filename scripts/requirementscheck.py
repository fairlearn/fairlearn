# Simple script which finds all files matching requirements*.txt in a given directory
# and makes sure that they match up to '>=' vs '=='

import argparse
# import importlib.util
import os
import re
import requirements
import sys


desc = "Checks setup.py and pip requirements files for consistency"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('directory')

args = parser.parse_args()

target_dir = os.path.abspath(args.directory)

print("Examining {0}".format(target_dir))

file_pattern = r"requirements.*\.txt"

files = [f for f in os.listdir(target_dir) if re.match(file_pattern, f)]

print("Found requirements files: {0}".format(files))

dependencies = {}
for f in files:
    print("Parsing file: {0}".format(f))
    target_file = os.path.join(target_dir, f)
    dep_dict = {}
    with open(f, 'r') as fd:
        for req in requirements.parse(fd):
            req_name = req.name
            if len(req.specs) > 1:
                print("Spec too complicated", file=sys.stderr)
                sys.exit(2)
            if len(req.specs) == 1:
                dep_dict[req_name] = req.specs[0][1]
            else:
                # Nothing specified so add a fake version
                dep_dict[req_name] = -1

    dependencies[f] = dep_dict
print("All requirements files parsed")

truth_file = list(dependencies.keys())[0]
print("Taking {0} as ground truth".format(truth_file))
truth_deps = dependencies[truth_file]
print()

found_mismatch = False
for f, d in dependencies.items():
    if f == truth_file:
        continue
    print("Examining {0}".format(f))
    if sorted(d.keys()) != sorted(truth_deps.keys()):
        print(" Dependency list does not match!")
        found_mismatch = True
    else:
        for dep, ver in truth_deps.items():
            if ver != d[dep]:
                print("  Mismatched versions for {0}".format(dep))
                found_mismatch = True
    print("Examination of {0} completed".format(f))
    print()

# spec = importlib.util.spec_from_file_location("setup", os.path.join(target_dir, "setup.py"))
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)
# print(module)

if found_mismatch:
    print("##[error]Found mismatches", file=sys.stderr)
    sys.exit(1)
else:
    sys.exit(0)
