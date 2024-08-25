import os
import sys
import pyseldonlib

sys.path.insert(0, os.path.abspath('../pyseldonlib'))

project = 'pyseldonlib'
copyright = '2024, PySeldon Developers'
author = 'Amrita Goswami, Daivik Karbhari, Moritz Sallermann, Rohit Goswami'
release = '1.0.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.napoleon', # from https://stackoverflow.com/a/66930447/22039471
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    # 'nbsphinx',
    'sphinxcontrib.bibtex',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx.ext.viewcode',
    "numpydoc",
]
bibtex_bibfiles = [
  'refs.bib',
  ]

jupyter_execute_notebooks = "auto"
autosummary_generate = True
numpydoc_show_class_members = False 

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'inherited-members': True,
    'show-inheritance': True,
}

templates_path = ['source/_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_title = "PySeldon"
html_logo = "../res/logotext.png"
html_favicon = "../res/favicon.ico"

html_theme_options = {
  "show_toc_level": 2,
    "icon_links": [
        {"name": "Home Page", "url": "https://github.com/seldon-code/pyseldonlib", "icon": "fas fa-home"},
        {
            "name": "GitHub",
            "url": "https://github.com/seldon-code/pyseldonlib",
            "icon": "fab fa-github-square",
        },
    ],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "pygments_light_style": "tango",
   "pygments_dark_style": "monokai",
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
    "index": [],
    "install": [],
    "quickstart": [],
    "contributing": [],
    "LICENSE": [],
    "examples/index": [],
}