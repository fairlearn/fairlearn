# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import subprocess
import time


def build_package():
    print('removing build directory')
    shutil.rmtree("build", True)
    print('removing fairlearn.egg-info')
    shutil.rmtree("fairlearn.egg-info", True)
    print('removing dist directory')
    shutil.rmtree("dist", True)

    print('running python setup.py bdist_wheel')
    subprocess.Popen(["python", "setup.py", "bdist_wheel"], cwd=os.getcwd(), shell=True).wait()
    for root, dirs, files in os.walk("dist"):
        for file in files:
            if file.endswith(".whl"):
                print("Found wheel {}".format(file))
                # change wheel name to be unique for every run
                src = os.path.join("dist", file)
                dst = os.path.join("dist", "fairlearn-v{}-py3-none-any.whl"
                                   .format(time.time()))
                shutil.copy(src, dst)
                return dst

    raise Exception("Couldn't find wheel file.")
