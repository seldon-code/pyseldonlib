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

```{figure} source/_static/res/pyseldon1.png
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
| [Installing](source/install.md) |
[User Guide](source/user-guide/user-guide.md) |
[API reference](source/api-reference/api-reference.md) |
[Contributing](source/contributing.md) |

```{toctree}
:maxdepth: 1
:caption: Contents


source/install.md
User Guide <source/user-guide/user-guide.md>
API Reference <source/api-reference/api-reference.md>
Examples <source/examples.md>
Contributing <source/contributing.md>
License <source/LICENSE>
```
source/api-reference/api-reference