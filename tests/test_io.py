import json
import os
import subprocess

import h5py
import nrrd
import numpy as np
import pytest
import tifffile

import sand_atlas.io


@pytest.mark.unit
def test_save_and_load_tiff_round_trip(tmp_path):
    data = np.arange(27, dtype=np.uint16).reshape(3, 3, 3)
    filename = tmp_path / "volume.tif"

    sand_atlas.io.save_data(data, str(filename), microns_per_voxel=2.0)
    loaded = sand_atlas.io.load_data(str(filename))

    assert np.array_equal(np.asarray(loaded), data)


@pytest.mark.unit
def test_save_and_load_npz_round_trip(tmp_path):
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    filename = tmp_path / "volume.npz"

    sand_atlas.io.save_data(data, str(filename))
    loaded = sand_atlas.io.load_data(str(filename))

    assert np.array_equal(loaded, data)


@pytest.mark.unit
def test_load_nrrd_returns_written_array(tmp_path):
    data = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    filename = tmp_path / "volume.nrrd"
    nrrd.write(str(filename), data)

    loaded = sand_atlas.io.load_data(str(filename))

    assert np.array_equal(loaded, data)


@pytest.mark.unit
def test_load_h5_reads_first_dataset(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    filename = tmp_path / "volume.h5"
    with h5py.File(filename, "w") as handle:
        handle.create_dataset("arr_0", data=data)

    loaded = sand_atlas.io.load_data(str(filename))

    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, data)


@pytest.mark.unit
def test_convert_loads_and_saves_between_formats(tmp_path):
    data = np.arange(8, dtype=np.uint8).reshape(2, 2, 2)
    input_filename = tmp_path / "input.npz"
    output_filename = tmp_path / "output.tif"
    np.savez(input_filename, data)

    sand_atlas.io.convert(str(input_filename), str(output_filename))
    loaded = tifffile.imread(output_filename)

    assert np.array_equal(loaded, data)


@pytest.mark.unit
def test_load_json_reads_expected_keys(tmp_path):
    filename = tmp_path / "metadata.json"
    payload = {"microns_per_pixel": "3.0", "URI": "example"}
    filename.write_text(json.dumps(payload))

    assert sand_atlas.io.load_json(str(filename)) == payload


@pytest.mark.unit
def test_find_blender_prefers_path_lookup(monkeypatch):
    monkeypatch.setattr(sand_atlas.io.shutil, "which", lambda _: "/custom/bin/blender")

    assert sand_atlas.io.find_blender() == "/custom/bin/blender"


@pytest.mark.unit
def test_add_to_path_appends_blender_directory(monkeypatch):
    original_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", "/usr/bin")

    sand_atlas.io.add_to_path("/Applications/Blender.app/Contents/MacOS/Blender")

    assert os.environ["PATH"].endswith("/Applications/Blender.app/Contents/MacOS")
    monkeypatch.setenv("PATH", original_path)


@pytest.mark.unit
def test_check_blender_command_exits_when_blender_missing(monkeypatch):
    monkeypatch.setattr(sand_atlas.io, "find_blender", lambda: None)

    with pytest.raises(SystemExit, match="1"):
        sand_atlas.io.check_blender_command()


@pytest.mark.unit
def test_check_ffmpeg_command_runs_version_probe(monkeypatch):
    recorded = {}

    def fake_run(command, check, stdout, stderr):
        recorded["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sand_atlas.io.subprocess, "run", fake_run)

    sand_atlas.io.check_ffmpeg_command()

    assert recorded["command"] == ["ffmpeg", "-version"]
