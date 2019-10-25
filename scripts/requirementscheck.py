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
    target_file = os.path.join(target_dir, f)
    dep_dict = {}
    with open(f, 'r') as fd:
        for req in requirements.parse(fd):
            req_name = req.name
            assert len(req.specs) <= 1
            if len(req.specs) == 1:
                dep_dict[req_name] = req.specs[0][1]

    dependencies[f] = dep_dict

truth_file = list(dependencies.keys())[0]
print("Taking {0} as ground truth".format(truth_file))
truth_deps = dependencies[truth_file]

found_mismatch = False
for f, d in dependencies.items():
    if f == truth_file:
        continue
    print("Examining {0}".format(f))
    if sorted(d.keys()) != sorted(truth_deps.keys()):
        print(" Dependency list does not match!", file=sys.stderr)
        found_mismatch = True
    else:
        for dep, ver in truth_deps.items():
            if ver != d[dep]:
                print("  Mismatched versions for {0}".format(dep), file=sys.stderr)
                found_mismatch = True

# spec = importlib.util.spec_from_file_location("setup", os.path.join(target_dir, "setup.py"))
# module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(module)
# print(module)

if found_mismatch:
    sys.exit(1)
else:
    sys.exit(0)
