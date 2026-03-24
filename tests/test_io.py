import json
import os
import subprocess
import sys
import types

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
def test_load_raw_requires_dimensions(tmp_path):
    filename = tmp_path / "volume.raw"
    filename.write_bytes(b"\x00\x01")

    with pytest.raises(ValueError, match="nx, ny, nz"):
        sand_atlas.io.load_data(str(filename))


@pytest.mark.unit
def test_load_raw_returns_memmap_when_dimensions_provided(tmp_path):
    filename = tmp_path / "volume.raw"
    np.arange(8, dtype=np.uint8).tofile(filename)

    loaded = sand_atlas.io.load_data(str(filename), nx=2, ny=2, nz=2)

    assert isinstance(loaded, np.memmap)
    assert np.array_equal(np.asarray(loaded), np.arange(8, dtype=np.uint8))


@pytest.mark.unit
def test_load_mic_requires_dimensions(tmp_path):
    filename = tmp_path / "volume.mic"
    filename.write_text("0\n1\n")

    with pytest.raises(ValueError, match="nx, ny, nz"):
        sand_atlas.io.load_data(str(filename))


@pytest.mark.unit
def test_load_mic_reshapes_and_flips_j_axis(tmp_path):
    filename = tmp_path / "volume.mic"
    values = "\n".join(str(index) for index in range(8))
    filename.write_text(values)

    loaded = sand_atlas.io.load_data(str(filename), nx=2, ny=2, nz=2)

    expected = np.arange(8, dtype=np.uint8).reshape((2, 2, 2))
    expected = np.flip(expected, axis=1)
    assert np.array_equal(loaded, expected)


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
def test_save_raw_round_trip(tmp_path):
    data = np.arange(6, dtype=np.uint8)
    filename = tmp_path / "volume.raw"

    sand_atlas.io.save_data(data, str(filename))

    assert np.array_equal(np.fromfile(filename, dtype=np.uint8), data)


@pytest.mark.unit
def test_save_h5_round_trip(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    filename = tmp_path / "volume.h5"

    sand_atlas.io.save_data(data, str(filename))

    with h5py.File(filename, "r") as handle:
        assert np.array_equal(handle["arr_0"][()], data)


@pytest.mark.unit
def test_save_neuroglancer_requires_cloudvolume(tmp_path):
    filename = tmp_path / "volume.neuroglancer"
    data = np.ones((2, 2, 2), dtype=np.uint8)

    sys.modules.pop("cloudvolume", None)

    with pytest.raises(ImportError, match="cloud-volume"):
        sand_atlas.io.save_data(data, str(filename))


@pytest.mark.unit
def test_save_neuroglancer_writes_volume_with_cloudvolume(monkeypatch, tmp_path):
    filename = tmp_path / "volume.neuroglancer"
    data = np.ones((2, 3, 4), dtype=np.uint8)
    recorded = {}

    class FakeCloudVolumeInstance:
        def commit_info(self):
            recorded["committed"] = True

        def __setitem__(self, key, value):
            recorded["key"] = key
            recorded["value"] = value.copy()

    class FakeCloudVolume:
        @staticmethod
        def create_new_info(**kwargs):
            recorded["info"] = kwargs
            return {"created": True}

        def __new__(cls, path, info, compress=False):
            recorded["path"] = path
            recorded["instance_info"] = info
            recorded["compress"] = compress
            return FakeCloudVolumeInstance()

    monkeypatch.setitem(sys.modules, "cloudvolume", types.SimpleNamespace(CloudVolume=FakeCloudVolume))

    sand_atlas.io.save_data(data, str(filename))

    assert recorded["path"].endswith("/neuroglancer")
    assert recorded["info"]["layer_type"] == "segmentation"
    assert recorded["info"]["volume_size"] == data.shape
    assert recorded["key"] == (slice(None, None, None), slice(None, None, None), slice(None, None, None))
    assert np.array_equal(recorded["value"], data)
    assert recorded["committed"] is True


@pytest.mark.unit
def test_find_blender_prefers_path_lookup(monkeypatch):
    monkeypatch.setattr(sand_atlas.io.shutil, "which", lambda _: "/custom/bin/blender")

    assert sand_atlas.io.find_blender() == "/custom/bin/blender"


@pytest.mark.unit
def test_find_blender_checks_platform_specific_paths(monkeypatch):
    monkeypatch.setattr(sand_atlas.io.shutil, "which", lambda _: None)
    monkeypatch.setattr(sand_atlas.io.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        sand_atlas.io.os.path,
        "exists",
        lambda path: path == "/Applications/Blender.app/Contents/MacOS/Blender",
    )

    assert sand_atlas.io.find_blender() == "/Applications/Blender.app/Contents/MacOS/Blender"


@pytest.mark.unit
def test_find_blender_returns_none_when_no_candidate_exists(monkeypatch):
    monkeypatch.setattr(sand_atlas.io.shutil, "which", lambda _: None)
    monkeypatch.setattr(sand_atlas.io.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sand_atlas.io.os.path, "exists", lambda path: False)

    assert sand_atlas.io.find_blender() is None


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
def test_check_blender_command_probes_version_when_blender_found(monkeypatch):
    recorded = {}

    monkeypatch.setattr(sand_atlas.io, "find_blender", lambda: "/custom/bin/blender")
    monkeypatch.setattr(sand_atlas.io, "add_to_path", lambda blender_path: recorded.update({"path": blender_path}))

    def fake_run(command, check, stdout, stderr):
        recorded["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sand_atlas.io.subprocess, "run", fake_run)

    sand_atlas.io.check_blender_command()

    assert recorded["path"] == "/custom/bin/blender"
    assert recorded["command"] == ["blender", "--version"]


@pytest.mark.unit
def test_check_blender_command_exits_on_called_process_error(monkeypatch):
    monkeypatch.setattr(sand_atlas.io, "find_blender", lambda: "/custom/bin/blender")
    monkeypatch.setattr(sand_atlas.io, "add_to_path", lambda blender_path: None)

    def fake_run(command, check, stdout, stderr):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(sand_atlas.io.subprocess, "run", fake_run)

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


@pytest.mark.unit
def test_check_ffmpeg_command_exits_on_called_process_error(monkeypatch):
    def fake_run(command, check, stdout, stderr):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(sand_atlas.io.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="1"):
        sand_atlas.io.check_ffmpeg_command()


@pytest.mark.unit
def test_load_data_rejects_unsupported_extension(tmp_path):
    filename = tmp_path / "volume.xyz"
    filename.write_text("unsupported")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        sand_atlas.io.load_data(str(filename))


@pytest.mark.unit
def test_save_data_rejects_unsupported_extension(tmp_path):
    filename = tmp_path / "volume.xyz"

    with pytest.raises(ValueError, match="Unsupported file extension"):
        sand_atlas.io.save_data(np.zeros((2, 2), dtype=np.uint8), str(filename))


@pytest.mark.feature
def test_make_zips_creates_expected_archive_commands(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    commands = []

    monkeypatch.setattr(sand_atlas.io.os.path, "exists", lambda path: path == str(output_dir))
    monkeypatch.setattr(sand_atlas.io.os, "makedirs", lambda path: commands.append(f"mkdir {path}"))
    monkeypatch.setattr(sand_atlas.io.os, "system", lambda command: commands.append(command) or 0)

    sand_atlas.io.make_zips(str(data_dir), str(output_dir))

    assert commands[0] == f"rm -rf {output_dir}/*.zip"
    assert f"zip -j {output_dir}/meshes_ORIGINAL.zip {data_dir}/stl_ORIGINAL/*.stl" in commands
    assert f"cp {data_dir}/stl_100/particle_00001.stl {output_dir}/ref_mesh_100.stl" in commands
    assert f"zip -j {output_dir}/multisphere_3.zip {data_dir}/multisphere/quality_3/*.csv" in commands
    assert f"cp {data_dir}/multisphere/quality_5/particle_00001.csv {output_dir}/ref_multisphere_5.csv" in commands
    assert f"zip -j {output_dir}/level_sets.zip {data_dir}/vdb/*.vdb" in commands
    assert f"zip -j {output_dir}/level_sets_YADE.zip {data_dir}/yade/*.npy" in commands


@pytest.mark.feature
def test_make_zips_creates_output_directory_when_missing(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr(sand_atlas.io.os.path, "exists", lambda path: False)
    monkeypatch.setattr(sand_atlas.io.os, "makedirs", lambda path: commands.append(("mkdir", path)))
    monkeypatch.setattr(sand_atlas.io.os, "system", lambda command: commands.append(("cmd", command)) or 0)

    sand_atlas.io.make_zips(str(tmp_path / "data"), str(tmp_path / "output"))

    assert commands[0] == ("mkdir", str(tmp_path / "output"))
