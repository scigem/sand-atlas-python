import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
def test_full_analysis_generates_label_and_outputs_artifacts(monkeypatch, tmp_path, import_with_fake_spam):
    pipeline = import_with_fake_spam()
    monkeypatch.chdir(tmp_path)

    json_path = tmp_path / "sample.json"
    json_path.write_text('{"microns_per_pixel": "2.0"}')

    raw_data = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    generated_label = np.ones((3, 3, 3), dtype=int)
    saved_paths = []
    events = []

    monkeypatch.setattr(pipeline.sand_atlas.io, "check_blender_command", lambda: events.append("check_blender"))
    monkeypatch.setattr(pipeline.sand_atlas.io, "check_ffmpeg_command", lambda: events.append("check_ffmpeg"))
    monkeypatch.setattr(pipeline.sand_atlas.io, "load_json", lambda path: {"microns_per_pixel": "2.0"})
    monkeypatch.setattr(pipeline.sand_atlas.io, "load_data", lambda path: raw_data)
    monkeypatch.setattr(pipeline, "gray_to_bw", lambda data, threshold, blur: events.append("gray_to_bw") or (data > 0))
    monkeypatch.setattr(
        pipeline,
        "label_binary_data",
        lambda binary_data: events.append("label_binary_data") or generated_label,
    )

    def fake_save_data(data, filename, microns_per_voxel=None):
        saved_paths.append(filename)
        output_path = tmp_path / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"saved")

    monkeypatch.setattr(pipeline.sand_atlas.io, "save_data", fake_save_data)
    monkeypatch.setattr(
        pipeline,
        "labelled_image_to_mesh",
        lambda labelled_data, sand_type, microns_per_voxel, output_dir, debug=False: events.append(
            ("mesh", sand_type, microns_per_voxel, output_dir, int(labelled_data.max()))
        ),
    )
    monkeypatch.setattr(
        pipeline.sand_atlas.io,
        "make_zips",
        lambda data_foldername, output_foldername: events.append(("zips", data_foldername, output_foldername)),
    )
    monkeypatch.setattr(
        pipeline,
        "get_particle_properties",
        lambda labelled_data, microns_per_voxel: pd.DataFrame({"Volume (µm³)": [12.0]}, index=[1]),
    )
    monkeypatch.setattr(
        pipeline.sand_atlas.video,
        "make_website_video",
        lambda stl_foldername, output_foldername: events.append(("video", stl_foldername, output_foldername)),
    )

    pipeline.full_analysis(str(json_path), raw_data_filename="raw.tif")

    assert (tmp_path / "output/sample/sample.csv").exists()
    assert "output/sample/upload/sample-labelled.tif" in saved_paths
    assert "output/sample/upload/sample-raw.tif" in saved_paths
    assert "gray_to_bw" in events
    assert "label_binary_data" in events
    assert ("mesh", "sample", 2.0, "output/sample", 1) in events
    assert ("zips", "output/sample", "output/sample/upload/") in events
    assert ("video", "output/sample/stl_ORIGINAL", "output/sample/upload/") in events


@pytest.mark.integration
def test_full_analysis_uses_existing_label_and_skips_raw_save(monkeypatch, tmp_path, import_with_fake_spam):
    pipeline = import_with_fake_spam()
    monkeypatch.chdir(tmp_path)

    json_path = tmp_path / "sample.json"
    label_path = tmp_path / "existing_label.tif"
    json_path.write_text('{"microns_per_pixel": "3.0"}')
    label_path.write_bytes(b"label")

    labelled = np.ones((3, 3, 3), dtype=int)
    saved_paths = []
    events = []

    monkeypatch.setattr(pipeline.sand_atlas.io, "check_blender_command", lambda: None)
    monkeypatch.setattr(pipeline.sand_atlas.io, "check_ffmpeg_command", lambda: None)
    monkeypatch.setattr(pipeline.sand_atlas.io, "load_json", lambda path: {"microns_per_pixel": "3.0"})
    monkeypatch.setattr(pipeline.sand_atlas.io, "load_data", lambda path: labelled)
    monkeypatch.setattr(
        pipeline,
        "gray_to_bw",
        lambda data, threshold, blur: (_ for _ in ()).throw(AssertionError("gray_to_bw should not run")),
    )
    monkeypatch.setattr(
        pipeline,
        "label_binary_data",
        lambda binary_data: (_ for _ in ()).throw(AssertionError("label_binary_data should not run")),
    )

    def fake_save_data(data, filename, microns_per_voxel=None):
        if data is None:
            raise AssertionError("save_data should not be called with raw_data=None")
        saved_paths.append(filename)
        output_path = tmp_path / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"saved")

    monkeypatch.setattr(pipeline.sand_atlas.io, "save_data", fake_save_data)
    monkeypatch.setattr(
        pipeline,
        "labelled_image_to_mesh",
        lambda labelled_data, sand_type, microns_per_voxel, output_dir, debug=False: events.append("mesh"),
    )
    monkeypatch.setattr(pipeline.sand_atlas.io, "make_zips", lambda *args: events.append("zips"))
    monkeypatch.setattr(
        pipeline,
        "get_particle_properties",
        lambda labelled_data, microns_per_voxel: pd.DataFrame({"Volume (µm³)": [5.0]}, index=[1]),
    )
    monkeypatch.setattr(pipeline.sand_atlas.video, "make_website_video", lambda *args: events.append("video"))

    pipeline.full_analysis(str(json_path), labelled_data_filename=str(label_path))

    assert (tmp_path / "output/sample/sample.csv").exists()
    assert "output/sample/upload/sample-labelled.tif" in saved_paths
    assert "output/sample/upload/sample-raw.tif" not in saved_paths
    assert events == ["mesh", "zips", "video"]
