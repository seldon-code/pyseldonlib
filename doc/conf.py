project = 'PySeldon'
copyright = '2024, Amrita Goswami, Rohit Goswami, Moritz Sallermann'
author = 'Daivik Karbhari'
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'myst_parser',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    "sphinxcontrib.autodoc_pydantic",
    "autodoc2",
]

templates_path = ['source/_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_book_theme'
html_title = "PySeldon"
html_static_path = ['source/_static']

html_theme_options = {
    "repository_url": "https://github.com/User-DK/pyseldon/tree/develop", #to do change url
    "use_repository_button": True,
    "use_fullscreen_button": False,
    "use_download_button": False,

}

# # -- Path setup --------------------------------------------------------------
# sys.path.insert(0, os.path.abspath('../pyseldon'))

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


# -- autodoc2 options --------------------------------------------------------
autodoc2_render_plugin = "myst"
autodoc2_packages = [
    {
        "path": "../src",
        # "auto_mode": False,
    },
]

# # -- sphinx_favicon options -----------------------------------------------
# favicons = [
#     "favicons/favicon-16x16.png",
#     "favicons/favicon-32x32.png",
#     "favicons/favicon.ico",
#     "favicons/android-chrome-192x192.png",
#     "favicons/android-chrome-512x512.png",
#     "favicons/apple-touch-icon.png",
#     "favicons/site.webmanifest",
# ]