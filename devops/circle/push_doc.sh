#!/bin/bash
# This script is meant to be called in the "deploy" step defined in
# circle.yml. See https://circleci.com/docs/ for more details.
# The behavior of the script is controlled by environment variable defined
# in the circle.yml in the top level folder of the project.

set -ex

if [ -z $CIRCLE_PROJECT_USERNAME ];
then USERNAME="fairlearn-ci";
else USERNAME=$CIRCLE_PROJECT_USERNAME;
fi

DOC_REPO="fairlearn.github.io"
GENERATED_DOC_DIR=$1

if [[ -z "$GENERATED_DOC_DIR" ]]; then
    echo "Need to pass directory of the generated doc as argument"
    echo "Usage: $0 <generated_doc_dir>"
    exit 1
fi

# Absolute path needed because we use cd further down in this script
GENERATED_DOC_DIR=$(readlink -f $GENERATED_DOC_DIR)

MSG="Pushing the docs to . for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

cd $HOME
if [ ! -d $DOC_REPO ];
then git clone --depth 1 --no-checkout "git@github.com:fairlearn/"$DOC_REPO".git";
fi
cd $DOC_REPO

cp -R "$GENERATED_DOC_DIR"/* .
git config user.email "adrin.jalali@gmail.com"
git config user.name $USERNAME
git config push.default matching
git add -f .
git commit -m "$MSG" $dir
git push
echo $MSG
