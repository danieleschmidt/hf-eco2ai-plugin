"""Sphinx configuration for HF Eco2AI Plugin documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "HF Eco2AI Plugin"
copyright = "2025, Daniel Schmidt"
author = "Daniel Schmidt"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "transformers": ("https://huggingface.co/docs/transformers", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True