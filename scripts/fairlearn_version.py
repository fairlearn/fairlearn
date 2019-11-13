# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Echos the current fairlearn version to stdout for use in the
# release pipelines

# This is meant to be run from an environment where fairlearn has
# not been pip installed, so have to fiddle with the path

import os
import sys

sys.path.insert(0, os.getcwd())

import fairlearn  # noqa: E402

print(fairlearn.__version__)
