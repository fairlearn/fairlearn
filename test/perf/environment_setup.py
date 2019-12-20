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

    # perf tests don't need dashboard, so we're faking the required files
    required_files = [
        os.path.join(os.getcwd(), 'fairlearn', 'widget', 'static', 'extension.js'),
        os.path.join(os.getcwd(), 'fairlearn', 'widget', 'static', 'extension.js.map'),
        os.path.join(os.getcwd(), 'fairlearn', 'widget', 'static', 'index.js'),
        os.path.join(os.getcwd(), 'fairlearn', 'widget', 'static', 'index.js.map'),
        os.path.join(os.getcwd(), 'fairlearn', 'widget', 'js', 'fairlearn_widget', 'labextension', 'fairlearn-widget-0.1.0.tgz')
    ]
    for file_path in required_files:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(file_path):
            with open(file_path, 'w') as new_file:
                new_file.write('')

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
