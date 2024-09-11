# Sand Atlas
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation here](https://scigem.github.io/sand-atlas-python/build/html/index.html), or compile it yourself following the details below.

## Installation
Can be installed via `pip install sand-atlas`. If installing from github, try cloning and then running:
```
pip install -e .
```
If you make any changes to the source code, re-run those two lines to have your changes reflected in your installed package.

## Documentation

We use `sphinx` to manage the docs. Update documentation with:
```
cd docs
make html
```
Once these are built, you can commit and push the changes to github to have them refreshed on github pages. You can also view them locally.