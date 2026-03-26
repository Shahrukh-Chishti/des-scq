# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'des_scq'
copyright = '2026, Shahrukh Chishti, Carla Illmann'
author = 'Shahrukh Chishti, Carla Illmann'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os, sys
sys.path.insert(0, os.path.abspath('../..'))  # points to your package root

extensions = [
    'sphinx.ext.autodoc',       # reads docstrings automatically
    'sphinx.ext.napoleon',      # understands NumPy/Google style docstrings
    'sphinx.ext.viewcode',      # adds [source] links to every function
    'sphinx.ext.intersphinx',   # cross-links to numpy/torch docs
]

# Napoleon settings — your docstrings use NumPy style
napoleon_numpy_docstring   = True
napoleon_google_docstring  = False
napoleon_use_param         = True
napoleon_use_returns       = True

# autodoc settings
autodoc_member_order       = 'bysource'  # preserve order from the file
autodoc_typehints          = 'description'

# intersphinx — lets :class:`torch.Tensor` link to PyTorch docs
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'torch':  ('https://pytorch.org/docs/stable', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
