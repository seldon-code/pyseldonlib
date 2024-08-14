import os
import sys

sys.path.insert(0, os.path.abspath('../pyseldon'))

project = 'pyseldon'
copyright = '2024, Amrita Goswami, Moritz Sallermann, Rohit Goswami, Daivik Karbhari'
author = 'Amrita Goswami, Moritz Sallermann, Rohit Goswami, Daivik Karbhari'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.coverage',
    'nbsphinx',
    'myst_nb',
    'sphinx_copybutton',
    # 'sphinx-jsonschema',
    'sphinx-pydantic',
    "numpydoc",
]

autosummary_generate = True

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
        {"name": "Home Page", "url": "https://github.com/seldon-code/pyseldon", "icon": "fas fa-home"},
        {
            "name": "GitHub",
            "url": "https://github.com/seldon-code/pyseldon",
            "icon": "fab fa-github-square",
        },
    ],
    # "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "pygments_light_style": "tango",
   "pygments_dark_style": "monokai",
}

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
    "index": [],
    "install": [],
    "examples": [],
    "quickstart": [],
    "contributing": [],
    "LICENSE": [],
    "examples": [],
}

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]