# Sand Atlas
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation here](https://scigem.github.io/sand-atlas-python/), or compile it yourself following the details below.

## Installation
First, install `ffmpeg` and `blender` on your system. Both of these need to be available in your system path (so that running `ffmpeg` and `blender` in the command line works).

This package can then be installed via `pip install sand-atlas`. If installing from github, try cloning and then running:
```
pip install -e .
```
If you make any changes to the source code, re-run those two lines to have your changes reflected in your installed package.

## Usage

### Processing particles
If you are interested in reproducing a dataset that is available in the Sand Atlas, you can use the installed script `sand_atlas_process`, like so:

```
sand_atlas_process <path_to_json_file> --label <path_to_labelled_image> --raw <path_to_raw_image>
```

If you supply only the raw image, the script will attempt to label the image for you. If you supply the labelled image, the script will use that to generate the dataset. The `json` file contains the metadata for the dataset you are interested in. The `json` files for the datasets in the Sand Atlas are available [here](https://github.com/scigem/sand-atlas/tree/main/_data/sands).

### VDB files
VDB files are the industry standard for level set data. They can be viewed in recent versions of the open source software `blender`, via `Add -> Volume -> Import OpenVDB`. If you would like to use these files in a python environment that doesn't include the `pyopenvdb` package, you can use the `vdb_to_npy` script provided by the `sand_atlas` package to convert the VDB file to a numpy array:
```
vdb_to_npy <path_to_vdb_file>
```

This will produce a `.npy` file in the same directory where you run the script. You can then load this file in python with `np.load`:
```python
import numpy as np
data = np.load('path_to_npy_file.npy')
```

### Downloading particles from the python API
You can query the server to get a list of all of the available types of particles with
```
import sand_atlas
particles = sand_atlas.data.list()
print(particles)
```

If you would like to download all of the particles with a particular mesh quality to the current working directory, you do this via
```
import sand_atlas
sand_atlas.data.get_all('SAND-NAME', MESH-QUALITY)
```

If you would like to download a specific particle, you can do this via
```
import sand_atlas
sand_atlas.data.get_by_id('SAND-NAME', MESH-QUALITY, PARTICLE-ID)
```

## Documentation

We use `sphinx` to manage the docs. Update documentation with:
```
cd docs
make html
```
Once these are built, you can commit and push the changes to github to have them refreshed on github pages. You can also view them locally.