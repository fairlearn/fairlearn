# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import subprocess
import time

from azureml.core import Environment


def build_package():
    print('removing build directory')
    shutil.rmtree("build", True)
    print('removing fairlearn.egg-info')
    shutil.rmtree("fairlearn.egg-info", True)
    print('removing dist directory')
    shutil.rmtree("dist", True)

    widget_path = os.path.join(os.getcwd(), 'fairlearn', 'widget', 'js')
    os.chdir(widget_path)
    try:
        print('running yarn install')
        subprocess.Popen(["yarn", "install"], shell=True).wait()
        print('removing yarn build:all')
        subprocess.Popen(["yarn", "build:all"], shell=True).wait()
    except Exception as ex:
        print(ex.message)
    finally:
        os.chdir(os.path.join('..', '..', '..'))

    print('running python setup.py bdist_wheel')
    subprocess.Popen(["python", "setup.py", "bdist_wheel"], cwd=os.getcwd(), shell=True).wait()
    for root, dirs, files in os.walk("dist"):
        for file in files:
            if file.endswith(".whl"):
                print("Found wheel {}".format(file))
                # change wheel name to be unique for every run
                src = os.path.join("dist", file)
                dst = os.path.join("dist", "fairlearn-0.4.0a{}-py3-none-any.whl".format(time.time()))
                shutil.copy(src, dst)
                return dst

    raise Exception("Couldn't find wheel file.")
