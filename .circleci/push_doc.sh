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

if [ "$CIRCLE_BRANCH" = "main" ]
then
    dir=main
else
    # Strip off 'release/' from the beginning and '.X' from the end
    dir="${CIRCLE_BRANCH:8:-2}"
fi

MSG="Pushing the docs to $dir/ for branch: $CIRCLE_BRANCH, commit $CIRCLE_SHA1"

cd $HOME
if [ ! -d $DOC_REPO ];
then git clone --depth 1 --no-checkout "git@github.com:fairlearn/"$DOC_REPO".git";
fi
cd $DOC_REPO

# check if it's a new branch

echo $dir > .git/info/sparse-checkout
if ! git show HEAD:$dir >/dev/null
then
	# directory does not exist. Need to make it so sparse checkout works
	mkdir $dir
	touch $dir/index.html
	git add $dir
fi
git checkout main
git reset --hard origin/main
if [ -d $dir ]
then
	git rm -rf $dir/ && rm -rf $dir/
fi
cp -R $GENERATED_DOC_DIR $dir
touch .nojekyll

# If we're working with main, we should also update the link from the
# landing page on fairlearn.org
if [ "$CIRCLE_BRANCH" = "main" ]
then
    echo "Copying all the newly generated  files for the static landing page"
    ls $GENERATED_DOC_DIR/..
    #js
    cp $GENERATED_DOC_DIR/../landing_page.js js/
    #css
    cp $GENERATED_DOC_DIR/../landing_page_style.css css/
    #html
    cp $GENERATED_DOC_DIR/../index.html  .
    #fonts
    cp $GENERATED_DOC_DIR/../SourceCodePro-Bold.ttf fonts/
    cp $GENERATED_DOC_DIR/../SourceCodePro-Medium.ttf fonts/
    cp $GENERATED_DOC_DIR/../SourceCodePro-Regular.ttf fonts/
    #images
    cp $GENERATED_DOC_DIR/../Button_GH_Default.svg images/
    cp $GENERATED_DOC_DIR/../Button_GH_Hover.svg images/
    cp $GENERATED_DOC_DIR/../Discord_Color.svg images/
    cp $GENERATED_DOC_DIR/../Discord_Inverse.svg images/
    cp $GENERATED_DOC_DIR/../fairlearn_logo.svg images/
    cp $GENERATED_DOC_DIR/../fairlearn-favicon.ico images/
    cp $GENERATED_DOC_DIR/../fairlearn-meta-image.jpg images/
    cp $GENERATED_DOC_DIR/../Github_Color.svg images/
    cp $GENERATED_DOC_DIR/../Github_Inverse.svg images/
    cp $GENERATED_DOC_DIR/../Nav_Discord_Default.svg images/
    cp $GENERATED_DOC_DIR/../Nav_Discord_Hover.svg images/
    cp $GENERATED_DOC_DIR/../Nav_GH_Default.svg images/
    cp $GENERATED_DOC_DIR/../Nav_GH_Hover.svg images/
    cp $GENERATED_DOC_DIR/../Stack_Overflow_Color.svg images/
    cp $GENERATED_DOC_DIR/../Stack_Overflow_Inverse.svg images/
    cp $GENERATED_DOC_DIR/../Twitter_Color.svg images/
    cp $GENERATED_DOC_DIR/../Twitter_Inverse.svg images/


fi

echo "fairlearn.org" > CNAME
git config user.email "ci-build@fairlearn.org"
git config user.name $USERNAME
git config push.default matching
git add -f $dir/
git commit -m "$MSG" $dir
if [ "$CIRCLE_BRANCH" = "main" ]
then
    git add -f js/
    # May not have changes to js directory, so use --allow-empty
    git commit --allow-empty -m "$MSG" js/
fi
git push
echo $MSG
