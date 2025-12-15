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

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'ArrayRecord'
copyright = '2024, ArrayRecord Contributors'
author = 'ArrayRecord Contributors'

# The full version, including alpha/beta/rc tags
release = '0.8.2'
version = '0.8.2'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    'myst_parser',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# HTML title and other metadata
html_title = 'ArrayRecord Documentation'
html_short_title = 'ArrayRecord'

# Favicon and logo
# html_favicon = '_static/favicon.ico'
# html_logo = '_static/logo.png'

# Show source link
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'sidebar_hide_name': False,
    'light_css_variables': {
        'color-brand-primary': '#2980B9',
        'color-brand-content': '#2980B9',
        'color-admonition-background': 'transparent',
    },
    'dark_css_variables': {
        'color-brand-primary': '#79afd1',
        'color-brand-content': '#79afd1',
        'color-admonition-background': 'transparent',
    },
    'source_repository': 'https://github.com/bzantium/array_record/',
    'source_branch': 'main',
    'source_directory': 'docs/',
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'apache_beam': ('https://beam.apache.org/releases/pydoc/current/', None),
}

# MyST parser configuration
myst_enable_extensions = [
    'deflist',
    'tasklist',
    'colon_fence',
]

# Autosummary settings
autosummary_generate = True
autosummary_generate_overwrite = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
}

# Mock imports for modules that aren't available during doc build
autodoc_mock_imports = [
    'array_record',
    'array_record.python',
    'array_record.python.array_record_module',
    'array_record.python.array_record_data_source',
    'array_record.beam',
    'array_record.beam.arrayrecordio',
    'array_record.beam.dofns',
    'array_record.beam.pipelines',
    'apache_beam',
    'google.cloud',
    'tensorflow',
    'riegeli',
    'etils',
]

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': None,
}
