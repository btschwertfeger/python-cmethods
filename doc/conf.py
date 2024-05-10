# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# pylint: disable=invalid-name

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

"""This module is the configuration for the Sphinx documentation building process"""

import sys
from pathlib import Path

project = "python-cmethods"
copyright = "2023, Benjamin Thomas Schwertfeger"  # pylint: disable=redefined-builtin
author = "Benjamin Thomas Schwertfeger"

# to import the package
parent_directory = Path("..").resolve()
sys.path.insert(0, str(parent_directory))

# import links
rst_epilog = ""
# Read link all targets from file
with open("links.rst", encoding="utf-8") as f:
    rst_epilog += f.read()


def setup(app) -> None:  # noqa: ARG001
    from os.path import join
    from shutil import copyfile

    copyfile(join("..", "examples", "examples.ipynb"), "examples.ipynb")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "links.rst", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_context = {
    "display_github": True,
    "github_user": "btschwertfeger",
    "github_repo": "python-cmethods",
    "github_version": "master/doc/",
}
