# Load all of the global Astropy configuration

from __future__ import annotations

# STDLIB
import datetime
import pathlib
import sys
from importlib import import_module
from importlib.metadata import version as get_version

# THIRD PARTY
import tomli


def get_authors() -> set[str]:
    """Get author information from ``pyproject.toml``s.

    Returns
    -------
    set[str]
        The authors.
    """

    authors: set[str] = set()
    cfg = pathlib.Path(__file__).parent.parent / "pyproject.toml"

    with cfg.open("rb") as f:
        toml = tomli.load(f)

    project = dict(toml["project"])
    authors.update({d["name"] for d in project["authors"]})

    return authors


# -- General configuration ----------------------------------------------------

# By default, highlight as Python 3.
highlight_language = "python3"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# Sphinx extensions
extensions = [
    "sphinx.ext.doctest",
    "sphinx_automodapi.automodapi",
    "pytest_doctestplus.sphinx.doctestplus",
]

autosummary_generate = True

automodapi_toctreedirnm = "api"

# The reST default role (used for this markup: `text`) to use for all
# documents. Set to the "smart" one.
default_role = "obj"

# Class documentation should contain *both* the class docstring and
# the __init__ docstring
autoclass_content = "both"

# This is added to the end of RST files - a good place to put substitutions to
# be used globally.
rst_epilog = """
.. |NumPyOverloader| replace:: :class:`~overload_numpy.NumPyOverloader`
"""

# intersphinx
intersphinx_mapping = {
    "python": (
        "https://docs.python.org/3/",
        (None, "http://data.astropy.org/intersphinx/python3.inv"),
    ),
}

# Show / hide TODO blocks
todo_include_todos = True

doctest_global_setup = """
import sys

python_version = sys.version_info
"""


# -- NumpyDoc Configuration ------------------------

# Don't show summaries of the members in each class along with the
# class' docstring
numpydoc_show_class_members = True

# Whether to create cross-references for the parameter types in the
# Parameters, Other Parameters, Returns and Yields sections of the docstring.
numpydoc_xref_param_type = True

# Words not to cross-reference. Most likely, these are common words used in
# parameter type descriptions that may be confused for classes of the same
# name. This can be overwritten or modified in packages and is provided here for
# convenience.
numpydoc_xref_ignore = {
    "or",
    "of",
    "thereof",
    "default",
    "optional",
    "keyword-only",
    "instance",
    "type",
    "class",
    "subclass",
    "method",
}

# Mappings to fully qualified paths (or correct ReST references) for the
# aliases/shortcuts used when specifying the types of parameters.
# Numpy provides some defaults
# https://github.com/numpy/numpydoc/blob/b352cd7635f2ea7748722f410a31f937d92545cc/numpydoc/xref.py#L62-L94
numpydoc_xref_aliases = {
    # Python terms
    "function": ":term:`python:function`",
    "iterator": ":term:`python:iterator`",
    "mapping": ":term:`python:mapping`",
}

# -- Project information ------------------------------------------------------

# This does not *have* to match the package name, but typically does
project = "overload_numpy"
author = ", ".join(get_authors())
copyright = f"{datetime.datetime.now().year}, {author}"

import_module(project)
package = sys.modules[project]

# The short X.Y version.
version = get_version("overload_numpy").split("-", 1)[0]
# The full version, including alpha/beta/rc tags.
release = get_version("overload_numpy")


# -- Options for HTML output ---------------------------------------------------

html_theme = "pydata_sphinx_theme"

# html_logo = '_static/<X>.png'

# html_theme_options = {
#     "logo_link": "index",
#     "icon_links": [
#         {
#             "name": "GitHub",
#             "url": "",
#             "icon": "fab fa-github-square",
#         },
#     ],
# }

# Custom sidebar templates, maps document names to template names.
html_sidebars = {"**": ["search-field.html", "sidebar-nav-bs.html"]}

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = ''

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = str(docs_root / '_static' / 'X.ico')

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = ''

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
# html_title = "overload_numpy.[X]"

# Output file base name for HTML help builder.
htmlhelp_basename = project + "doc"

# Static files to copy after template files
html_static_path = ["_static"]
html_css_files = ["theme.css"]
