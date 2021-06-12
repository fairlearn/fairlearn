# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

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
import inspect
rootdir = os.path.join(os.getenv("SPHINX_MULTIVERSION_SOURCEDIR", default=os.getcwd()), "..")
sys.path.insert(0, rootdir)
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
[print(p) for p in sys.path]
print("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
import fairlearn  # noqa: E402
print(fairlearn.__version__)
print("================================")


# -- Project information -----------------------------------------------------

project = 'Fairlearn'
copyright = '2019, Microsoft Corporation and contributors.'
author = 'Microsoft and Fairlearn contributors'

# The full version, including alpha/beta/rc tags
release = fairlearn.__version__


def check_if_v046():
    """Check to see if current version being built is v0.4.6."""
    result = False

    if fairlearn.__version__ == "0.4.6":
        print("Detected 0.4.6 in fairlearn.__version__")
        result = True

    smv_name = os.getenv("SPHINX_MULTIVERSION_NAME")
    if smv_name is not None:
        print("Found SPHINX_MULTIVERSION_NAME: ", smv_name)
        result = smv_name == "v0.4.6"
    else:
        print("SPHINX_MULTIVERSION_NAME not in environment")

    return result


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'bokeh.sphinxext.bokeh_plot',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx_gallery.gen_gallery',
    'sphinx_multiversion'
]

intersphinx_mapping = {'python3': ('https://docs.python.org/3', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'sklearn': ('https://scikit-learn.org/stable/', None), }

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

master_doc = 'index'

# Multiversion settings

smv_tag_whitelist = r'^v0\.4\.6|^v0\.5\.\d|^v0\.6\.\d+$'
smv_branch_whitelist = r'^main$'

if check_if_v046():
    print("Current version is v0.4.6, will apply overrides")
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
    "logo_link": "https://fairlearn.org",

    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fairlearn/fairlearn",
            "icon": "fab fa-github",
        },
        {
            "name": "Twitter",
            "url": "https://twitter.com/fairlearn",
            "icon": "fab fa-twitter",
        },
        {
            "name": "StackOverflow",
            "url": "https://stackoverflow.com/questions/tagged/fairlearn",
            "icon": "fab fa-stack-overflow",
        },
        {
            "name": "Gitter",
            "url": "https://gitter.im/fairlearn/community",
            "icon": "fab fa-gitter",
        },
    ],

    "show_prev_next": False,

    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["footer-copyright"],
}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/images/fairlearn_full_color.png"

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
}

# If false, no index is generated.
html_use_index = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Use filename_pattern so that plot_adult_dataset is not
# included in the gallery, but its plot is available for
# the quickstart
sphinx_gallery_conf = {
    'examples_dirs': '../examples',
    'gallery_dirs': 'auto_examples',
    # pypandoc enables rst to md conversion in downloadable notebooks
    'pypandoc': True,
}

html_sidebars = {
    "**": ["version-sidebar.html", "search-field.html", "sidebar-nav-bs.html"],
}

# Auto-Doc Options
# ----------------

# Change the ordering of the member documentation
autodoc_member_order = 'groupwise'


# Linking Code
# ------------

# The following is used by sphinx.ext.linkcode to provide links to github
# based on pandas doc/source/conf.py
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    tag_or_branch = os.getenv("SPHINX_MULTIVERSION_NAME", default="main")
    fn = os.path.relpath(fn, start=os.path.dirname(fairlearn.__file__)).replace(os.sep, '/')
    return f"http://github.com/fairlearn/fairlearn/blob/{tag_or_branch}/fairlearn/{fn}{linespec}"


# -- LaTeX macros ------------------------------------------------------------

mathjax3_config = {
    "tex": {
        "macros": {
            "E": '{\\mathbb{E}}',
            "P": '{\\mathbb{P}}',
            "given": '\\mathbin{\\vert}'
        }
    }
}
