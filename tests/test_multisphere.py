import importlib
import sys
import types

import numpy as np
import pytest


def import_multisphere(monkeypatch, *, bounding_boxes=None, distance_field=None):
    fake_spam = types.ModuleType("spam")
    fake_label = types.ModuleType("spam.label")
    fake_filters = types.ModuleType("spam.filters")

    fake_label.boundingBoxes = bounding_boxes or (lambda labelled_data: {})
    fake_filters.distanceField = distance_field or (lambda image: image.astype(float))

    fake_spam.label = fake_label
    fake_spam.filters = fake_filters

    monkeypatch.setitem(sys.modules, "spam", fake_spam)
    monkeypatch.setitem(sys.modules, "spam.label", fake_label)
    monkeypatch.setitem(sys.modules, "spam.filters", fake_filters)
    sys.modules.pop("sand_atlas.multisphere", None)

    return importlib.import_module("sand_atlas.multisphere")


@pytest.mark.unit
def test_labelled_image_to_multipheres_calls_binary_to_clump_per_particle(monkeypatch, tmp_path):
    labelled = np.zeros((12, 12, 12), dtype=int)
    labelled[1:6, 1:6, 1:6] = 1
    labelled[6:11, 6:11, 6:11] = 2
    boxes = {
        1: (1, 6, 1, 6, 1, 6),
        2: (6, 11, 6, 11, 6, 11),
    }

    multisphere = import_multisphere(monkeypatch, bounding_boxes=lambda data: boxes)
    recorded = []

    monkeypatch.setattr(
        multisphere,
        "binary_to_clump",
        lambda image, microns_per_voxel, output_dir, filenamePrefix, **kwargs: recorded.append(
            {
                "shape": image.shape,
                "voxels": int(image.sum()),
                "microns_per_voxel": microns_per_voxel,
                "output_dir": output_dir,
                "filename": filenamePrefix,
                "kwargs": kwargs,
            }
        ),
    )
    monkeypatch.setattr(multisphere, "tqdm", lambda iterable, **kwargs: iterable)

    multisphere.labelled_image_to_multipheres(labelled, "sample-sand", 2.5, str(tmp_path), debug=True)

    assert tmp_path.exists()
    assert [call["filename"] for call in recorded] == ["particle_000001", "particle_000002"]
    assert all(call["shape"] == (5, 5, 5) for call in recorded)
    assert all(call["voxels"] == 125 for call in recorded)
    assert all(call["microns_per_voxel"] == 2.5 for call in recorded)
    assert all(call["kwargs"] == {"numpasses": 5, "debug": True, "save_all_passes": True} for call in recorded)


@pytest.mark.unit
def test_binary_to_clump_saves_scaled_spheres(monkeypatch, tmp_path):
    multisphere = import_multisphere(monkeypatch)
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1:3, 1:3, 1:3] = 1
    dt = np.zeros_like(image, dtype=float)
    dt[1, 1, 1] = 1.0
    saved = {}

    monkeypatch.setattr(multisphere, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(multisphere, "distanceField", lambda diff_img: dt)
    monkeypatch.setattr(multisphere, "peak_local_max", lambda *args, **kwargs: np.array([[1, 1, 1]]))
    monkeypatch.setattr(
        multisphere.np,
        "savetxt",
        lambda path, data, delimiter, header, comments: saved.update(
            {
                "path": path,
                "data": data.copy(),
                "delimiter": delimiter,
                "header": header,
                "comments": comments,
            }
        ),
    )

    multisphere.binary_to_clump(
        image,
        2.0,
        str(tmp_path),
        "particle_000001",
        numpasses=1,
        debug=False,
        save_all_passes=True,
    )

    assert saved["path"].endswith("multisphere/quality_1/particle_000001.csv")
    assert saved["data"].shape == (1, 4)
    assert saved["data"][0, 3] > 0
    assert saved["delimiter"] == ","
    assert saved["header"] == "x (microns),y (microns),z (microns),radius (microns)"
    assert saved["comments"] == ""


@pytest.mark.unit
def test_binary_to_clump_reports_when_no_spheres_are_found(monkeypatch, tmp_path, capsys):
    multisphere = import_multisphere(monkeypatch)
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1:3, 1:3, 1:3] = 1

    monkeypatch.setattr(multisphere, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(multisphere, "distanceField", lambda diff_img: np.zeros_like(diff_img, dtype=float))
    monkeypatch.setattr(multisphere, "peak_local_max", lambda *args, **kwargs: np.empty((0, 3), dtype=int))

    multisphere.binary_to_clump(
        image,
        1.0,
        str(tmp_path),
        "particle_000002",
        numpasses=1,
        debug=False,
        save_all_passes=True,
    )

    assert "No spheres found in pass 1. Skipping saving for this pass." in capsys.readouterr().out


@pytest.mark.unit
def test_binary_to_clump_debug_saves_plot(monkeypatch, tmp_path):
    multisphere = import_multisphere(monkeypatch)
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[1:3, 1:3, 1:3] = 1
    saved = {}

    class FakeAxes:
        def __init__(self):
            self.voxel_calls = 0

        def voxels(self, data, alpha=0.5):
            self.voxel_calls += 1

    class FakeFigure:
        def add_subplot(self, *args, **kwargs):
            return FakeAxes()

    monkeypatch.setattr(multisphere, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(multisphere, "distanceField", lambda diff_img: np.zeros_like(diff_img, dtype=float))
    monkeypatch.setattr(multisphere, "peak_local_max", lambda *args, **kwargs: np.empty((0, 3), dtype=int))
    monkeypatch.setattr(multisphere.plt, "figure", lambda: FakeFigure())
    monkeypatch.setattr(multisphere.plt, "savefig", lambda path: saved.update({"path": path}))

    multisphere.binary_to_clump(
        image,
        1.0,
        str(tmp_path),
        "particle_000003",
        numpasses=1,
        debug=True,
        save_all_passes=False,
    )

    assert saved["path"].endswith("multisphere/quality_1/particle_000003.png")


@pytest.mark.unit
def test_stl_to_clump_raises_when_no_stl_files_exist(monkeypatch):
    multisphere = import_multisphere(monkeypatch)
    monkeypatch.setattr(multisphere, "glob", lambda pattern: [])

    with pytest.raises(ValueError, match="No STL files found"):
        multisphere.stl_to_clump("/tmp/missing", 3)


@pytest.mark.unit
def test_stl_to_clump_generates_output_paths(monkeypatch, tmp_path):
    multisphere = import_multisphere(monkeypatch)
    input_dir = tmp_path / "meshes"
    input_dir.mkdir()
    stl_path = input_dir / "particle_00001.stl"
    stl_path.write_text("solid")
    recorded = {}

    monkeypatch.setattr(multisphere, "glob", lambda pattern: [str(stl_path)])
    monkeypatch.setattr(
        multisphere,
        "GenerateClump_Euclidean_3D",
        lambda stl_file, N, rMin, div, overlap, output, outputVTK, visualise: recorded.update(
            {
                "stl_file": stl_file,
                "N": N,
                "rMin": rMin,
                "div": div,
                "overlap": overlap,
                "output": output,
                "outputVTK": outputVTK,
                "visualise": visualise,
            }
        ),
        raising=False,
    )

    multisphere.stl_to_clump(str(input_dir), 7)

    assert recorded["stl_file"] == str(stl_path)
    assert recorded["N"] == 7
    assert recorded["rMin"] == 0
    assert recorded["div"] == 102
    assert recorded["overlap"] == 0.6
    assert recorded["output"].endswith("clumps/particle_00001.txt")
    assert recorded["outputVTK"].endswith("clumps/particle_00001.vtk")
    assert recorded["visualise"] is False


@pytest.mark.unit
def test_view_multisphere_draws_all_spheres(monkeypatch):
    multisphere = import_multisphere(monkeypatch)
    sphere_data = np.array(
        [
            [1.0, 2.0, 3.0, 0.5],
            [4.0, 5.0, 6.0, 1.0],
        ]
    )
    recorded = {"surface_calls": 0, "show": False, "tight_layout": False}

    class FakeAxes:
        def plot_surface(self, xs, ys, zs, color, alpha, linewidth, antialiased):
            recorded["surface_calls"] += 1

        def set_xlabel(self, label):
            recorded["xlabel"] = label

        def set_ylabel(self, label):
            recorded["ylabel"] = label

        def set_zlabel(self, label):
            recorded["zlabel"] = label

        def set_box_aspect(self, aspect):
            recorded["aspect"] = aspect

    class FakeFigure:
        def add_subplot(self, *args, **kwargs):
            return FakeAxes()

    monkeypatch.setattr(multisphere.np, "loadtxt", lambda filename, delimiter, skiprows: sphere_data)
    monkeypatch.setattr(multisphere.plt, "figure", lambda figsize: FakeFigure())
    monkeypatch.setattr(multisphere.plt, "tight_layout", lambda: recorded.update({"tight_layout": True}))
    monkeypatch.setattr(multisphere.plt, "show", lambda: recorded.update({"show": True}))

    multisphere.view_multisphere("particle.csv")

    assert recorded["surface_calls"] == 2
    assert recorded["xlabel"] == "X"
    assert recorded["ylabel"] == "Y"
    assert recorded["zlabel"] == "Z"
    assert recorded["tight_layout"] is True
    assert recorded["show"] is True
