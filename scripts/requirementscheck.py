"""Simple script for comparing two requirements files.

Given two requirements files, this script:
- Checks that they both contain the same list of packages
- Checks that the package versions match according to the following
  rules:
    - If no version is specified for one, it must not be specified
      for the other
    - If a version is specified, then the only valid comparators are
      '==' and '>='
    - If a version is specified in one file, then the same version
      must be specified in the other
The purposes of these comparisons is to ensure that our
`requirements.txt` and `requirements-fixed.txt` files are kept in
sync.

If any of the checks fail, then the script will write out information
about the failures (preprended for highlighting as an error in Azure
Dev Ops) and return 1 instead of 0.

At present there is no support for parsing `setup.py` and checking
the packages listed there.
"""
# Simple script which finds all files matching requirements*.txt in a given directory
# and makes sure that they match up to '>=' vs '=='

import argparse
import requirements
import sys

error_prefix = "##[error]"


def print_ado_error(msg):
    print("{0}{1}".format(error_prefix, msg))


def check_requirement_specs(requirement):
    """Ensures that the requirement is '=='
    or '>=' since we're interested in pinning
    lower bounds
    """
    specs_good = True

    if len(requirement.specs) > 1:
        msg = "Too many specs for {0}".format(requirement.name)
        print_ado_error(msg)
        specs_good = False
    elif len(requirement.specs) == 1:
        allowed = {">=", "=="}
        cmp = requirement.specs[0][0]
        if requirement.specs[0][0] not in allowed:
            msg = "Comparator {0} not allowed for {1}".format(cmp, requirement.name)
            print_ado_error(msg)
            specs_good = False

    return specs_good


def load_requirements_file(filename):
    """Loads the specified requirements file
    and checks that the specs are OK by themselves
    """

    print("Loading {0}".format(filename))

    good_requirements = True
    with open(filename, 'r') as fd:
        reqs = sorted(list(requirements.parse(fd)), key=lambda x: x.name)
        for r in reqs:
            good_requirements = good_requirements and check_requirement_specs(r)

    return good_requirements, reqs


def compare_specs(spec1, spec2):
    """Compare two specifications assuming that there
    is a maximum of one comparator. If present this is
    assumed to be == or >=
    """
    specs_match = True
    # Quick sanity check
    assert spec1.name == spec2.name

    have1 = len(spec1.specs) > 0
    have2 = len(spec2.specs) > 0

    if have1 != have2:
        msg = "Only one spec on {0}".format(spec1.name)
        print_ado_error(msg)
        specs_match = False
    elif not have1:
        # We have two empty lists
        pass
    else:
        if spec1.specs[0][1] != spec2.specs[0][1]:
            msg = "Version mismatch on {0}".format(spec1.name)
            print_ado_error(msg)
            specs_match = False

    return specs_match


def compare_requirements(req1, req2):
    """Compare the two requirements, assuming that
    check_requirement_specs has already been run and
    hence that we don't need to worry about the
    comparator
    """
    names1 = [x.name for x in req1]
    names2 = [x.name for x in req1]

    all_match = True
    if names1 != names2:
        msg = "List of requirement names does not match"
        print_ado_error(msg)
        return False
    else:
        all_match = True
        matches = []
        for i in range(len(names1)):
            # No need to check the spec again
            spec1 = req1[i]
            spec2 = req2[i]
            matches.append(compare_specs(spec1, spec2))
        all_match = all_match and any(matches)

    return all_match


desc = "Checks two requirements files for consistency"

parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--fixed')
parser.add_argument('--lowerbound')

args = parser.parse_args()

fixed_filename = args.fixed
lowerbound_filename = args.lowerbound

requirement_match_good = True

good_fixed, fixed_specs = load_requirements_file(fixed_filename)
good_lb, lb_specs = load_requirements_file(lowerbound_filename)

print()
print("Files loaded")
print()

requirement_match_good = good_fixed and good_lb

requirement_match_good = requirement_match_good and compare_requirements(fixed_specs, lb_specs)

if not requirement_match_good:
    print_ado_error("Found mismatches")
    sys.exit(1)
else:
    print("No mismatches found")
    sys.exit(0)
