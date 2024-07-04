---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: myst
  kernelspec:
    display_name: Python 3
    language: python
    name: python3

myst:
  substitutions:
    today: "2023-04-01"
    version: "1.0"
---

```{figure} _static/res/logo_text.png
---
align: center
width: 500px
alt: Pyseldon logo
---
```

# PySeldon documentation

**Date**: {{ today }} **Version**: {{ version }}

**PySeldon** is an open-source Library for Opinion Dynamics Simulation an Extension of the [Seldon](https://github.com/seldon-code/seldon) Framework

**Useful links**:
| Installing |
[User Guide](user-guide/user-guide.md) |
[API reference](api-reference/api-reference.md) |
[Contributing](contributing.md) |

```{toctree}
:maxdepth: 1
:caption: Contents


install.md
User Guide <user-guide/user-guide.md>
API Reference <api-reference/api-reference.md>
Examples <examples.md>
Contributing <contributing.md>
License <LICENSE>
```