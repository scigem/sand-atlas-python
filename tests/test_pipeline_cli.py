import json
import subprocess

import numpy as np
import pandas as pd
import pytest


@pytest.mark.feature
def test_full_analysis_script_requires_raw_or_label(monkeypatch, capsys, import_with_fake_spam, sample_metadata_json):
    pipeline = import_with_fake_spam()

    monkeypatch.setattr(
        pipeline.argparse.ArgumentParser,
        "parse_args",
        lambda self: pipeline.argparse.Namespace(
            json=str(sample_metadata_json),
            raw=None,
            label=None,
            threshold=None,
            blur=None,
            binning=None,
        ),
    )

    pipeline.full_analysis_script()

    captured = capsys.readouterr()
    assert "You must provide a raw data file" in captured.out


@pytest.mark.feature
def test_full_analysis_script_delegates_with_parsed_arguments(monkeypatch, import_with_fake_spam, sample_metadata_json):
    pipeline = import_with_fake_spam()
    recorded = {}

    monkeypatch.setattr(
        pipeline.argparse.ArgumentParser,
        "parse_args",
        lambda self: pipeline.argparse.Namespace(
            json=str(sample_metadata_json),
            raw="raw.tif",
            label="label.tif",
            threshold=42,
            blur=1.5,
            binning=3,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "full_analysis",
        lambda json_filename, **kwargs: recorded.update({"json": json_filename, **kwargs}),
    )

    pipeline.full_analysis_script()

    assert recorded == {
        "json": str(sample_metadata_json),
        "raw_data_filename": "raw.tif",
        "labelled_data_filename": "label.tif",
        "threshold": 42,
        "blur": 1.5,
        "binning": 3,
    }


@pytest.mark.feature
def test_properties_script_writes_output_csv(monkeypatch, tmp_path, import_with_fake_spam, sample_metadata_json):
    pipeline = import_with_fake_spam()
    output_path = tmp_path / "summary.csv"
    label_path = tmp_path / "labels.tif"
    label_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        pipeline.argparse.ArgumentParser,
        "parse_args",
        lambda self: pipeline.argparse.Namespace(
            json=str(sample_metadata_json),
            label=str(label_path),
            binning=None,
            output=str(output_path),
        ),
    )
    monkeypatch.setattr(pipeline.sand_atlas.io, "load_data", lambda _: np.ones((3, 3, 3), dtype=int))
    monkeypatch.setattr(
        pipeline,
        "get_particle_properties",
        lambda labelled_data, microns_per_voxel: pd.DataFrame({"Volume (µm³)": [12.5]}, index=[1]),
    )

    pipeline.properties_script()

    written = output_path.read_text()
    assert "Particle ID" in written
    assert "Volume (µm³)" in written
    assert "12.5" in written


@pytest.mark.feature
def test_clean_labels_script_delegates_to_clean_module(monkeypatch, import_with_fake_spam):
    pipeline = import_with_fake_spam()
    recorded = {}

    monkeypatch.setattr(
        pipeline.argparse.ArgumentParser,
        "parse_args",
        lambda self: pipeline.argparse.Namespace(label="labels.tif", num_processors=7, verbosity=2),
    )
    monkeypatch.setattr(
        pipeline.sand_atlas.clean,
        "clean_labels",
        lambda label, num_processors, verbosity: recorded.update(
            {"label": label, "num_processors": num_processors, "verbosity": verbosity}
        ),
    )

    pipeline.clean_labels_script()

    assert recorded == {"label": "labels.tif", "num_processors": 7, "verbosity": 2}


@pytest.mark.feature
def test_vdb_to_npy_invokes_blender_with_expected_arguments(monkeypatch, import_with_fake_spam):
    pipeline = import_with_fake_spam()
    recorded = {}

    monkeypatch.setattr(
        pipeline.argparse.ArgumentParser,
        "parse_args",
        lambda self: pipeline.argparse.Namespace(vdb_filename="sample.vdb"),
    )
    monkeypatch.setattr(pipeline.sand_atlas.io, "check_blender_command", lambda: None)
    monkeypatch.setattr(
        pipeline.sand_atlas.video,
        "resolve_path_for_blender",
        lambda script_relative_path: f"/abs/{script_relative_path}",
    )

    def fake_run(command):
        recorded["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    pipeline.vdb_to_npy()

    assert recorded["command"] == [
        "blender",
        "--background",
        "-noaudio",
        "--python",
        "/abs/blender_scripts/vdb_to_npy.py",
        "--",
        "sample.vdb",
    ]
