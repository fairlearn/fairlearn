# Azure DevOps Pipelines for Fairlearn

This directory contains YAML definitions of our Pipelines in Azure DevOps (ADO). These are [documented extensively online](https://docs.microsoft.com/en-us/azure/devops/pipelines/get-started/key-pipelines-concepts?view=azure-devops).

## Directory Layout

The current directory only contains the YAML files which correspond to ADO Pipelines.
They rely extensively on other YAML files in the `templates` directory.
As the name implies, these are included into the Pipelines via the templating mechanism provided by ADO.
Some of the templates in turn include other templates.

## The Pipelines

`code-coverage.yml` defines our code coverage build.
It is slightly different to the others, due to the need to instrument the test run and process and present the results afterwards.

`nightly-requirements-fixed.yml` runs an extra script on the `requirements.txt` file, which pins versions specified by a lower bound (i.e. it converts `>=` into `==` in the file).
This provides some measure of backwards compatibility checking.

`nightly.yml` runs tests against multiple Python versions on multiple platforms.
Due to the number of jobs spawned, it can take over an hour to run.

`PR-Gate.yml` is the main Pipeline triggered for pull requests.
It runs a subset of the tests in `nightly.yml`.

`pypi-release-new.yml` is a simplified Pipeline for releasing Fairlearn to PyPI.
At queue time, the user choses whether to target Test or Production PyPI.
The pipeline then:
1. Runs tests against the repository
1. Builds the Wheel
1. Runs tests against the Wheel
1. Uploads to PyPI (requires user approval)
1. Runs tests against the upload

`pypi-release.yml` is our original Pipeline for doing PyPI releases.
Rather than testing against the Wheel file, it first uploads to Test-PyPI, and then to Production PyPI.
In order to avoid 'burning' a version in Test PyPI (at the time the pipeline was created, PyPI automatically published
a package immediately on upload), the user must set a variable
`DEV_VERSION` at queue time to a unique integer.
The package uploaded to Test-PyPI then gets a `.dev[n]` suffix.
