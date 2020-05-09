# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
[print(p) for p in sys.path]
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
import fairlearn  # noqa: E402
print(fairlearn.__version__)
print("================================")


# -- Project information -----------------------------------------------------

project = 'Fairlearn'
copyright = '2019, Microsoft Corporation.'
author = 'Microsoft'

# The full version, including alpha/beta/rc tags
release = fairlearn._base_version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx_gallery.gen_gallery",
    "jupyter_sphinx"
]

intersphinx_mapping = {'python3': ('https://docs.python.org/3', None),
                       'sklearn': ('https://scikit-learn.org/stable/', None),
                       'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    # TODO: fork the pydata-sphinx-theme to integrate these with logo
    "external_links": [
        {"name": "Gitter", "url": "https://gitter.im/fairlearn/community"},
        {"name": "StackOverflow", "url": "https://stackoverflow.com/questions/tagged/fairlearn"}
    ],
    "github_url": "https://github.com/fairlearn/fairlearn",
    # "twitter_url": "https://twitter.com/fairlearn" TODO: start using this,
    "show_prev_next": False
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
# html_logo = "path/to/some/file.png" TODO: add logo

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    "css/cards.css"
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

sphinx_gallery_conf = {
     'examples_dirs': '../examples',
     'gallery_dirs': 'auto_examples',
}

# -- LaTeX macros ------------------------------------------------------------

mathjax_config = {
    "TeX": {
        "Macros": {
            "E": '{\\mathbb{E}}',
            "P": '{\\mathbb{P}}',
            "given": '\\mathbin{\\vert}'
            }
        }
    }
