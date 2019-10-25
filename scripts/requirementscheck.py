import argparse
import importlib.util
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


spec = importlib.util.spec_from_file_location("setup", os.path.join(target_dir, "setup.py"))
module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(module)

print(module)
