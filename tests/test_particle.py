import types

import numpy as np
import pytest


class DummyProgressBar:
    def __init__(self, *args, **kwargs):
        self.updates = []

    def start(self):
        return self

    def update(self, value):
        self.updates.append(value)

    def finish(self):
        return None


class FakePool:
    def __init__(self, processes):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, tasks):
        yield (2, 20)
        yield (1, 10)

    def close(self):
        return None

    def join(self):
        return None


@pytest.mark.unit
def test_compute_convex_volume_returns_zero_when_particle_has_too_few_points(import_with_fake_spam):
    particle = import_with_fake_spam(
        "sand_atlas.particle",
        spam_overrides={
            "getLabel": lambda lab, label, **kwargs: {"subvol": np.array([[[1, 0], [0, 0]], [[0, 1], [0, 1]]])},
        },
    )

    label, volume = particle.computeConvexVolume((1, np.zeros((2, 2, 2), dtype=int), None, None))

    assert label == 1
    assert volume == 0


@pytest.mark.unit
def test_compute_convex_volume_returns_mocked_hull_volume(monkeypatch, import_with_fake_spam):
    particle = import_with_fake_spam(
        "sand_atlas.particle",
        spam_overrides={
            "getLabel": lambda lab, label, **kwargs: {"subvol": np.ones((2, 2, 2), dtype=np.uint8)},
            "volumes": lambda lab: np.array([0, 8], dtype=float),
        },
    )

    monkeypatch.setattr(
        particle.scipy.spatial, "ConvexHull", lambda points: types.SimpleNamespace(vertices=[0, 1, 2, 3])
    )

    class FakeDelaunay:
        def __init__(self, vertices):
            self.vertices = vertices

        def find_simplex(self, coords):
            return np.zeros(len(coords), dtype=int)

    monkeypatch.setattr(particle.scipy.spatial, "Delaunay", FakeDelaunay)

    label, volume = particle.computeConvexVolume((1, np.zeros((2, 2, 2), dtype=int), None, None))

    assert label == 1
    assert volume == 8


@pytest.mark.unit
def test_convex_volume_collects_pool_results_in_label_order(monkeypatch, import_with_fake_spam):
    particle = import_with_fake_spam(
        "sand_atlas.particle",
        spam_overrides={
            "boundingBoxes": lambda lab: "boxes",
            "centresOfMass": lambda lab: "centres",
            "volumes": lambda lab: np.array([0, 5, 6], dtype=float),
        },
    )

    monkeypatch.setattr(particle.multiprocessing, "Pool", FakePool)
    monkeypatch.setattr(
        particle.progressbar,
        "ProgressBar",
        lambda *args, **kwargs: DummyProgressBar(),
    )
    monkeypatch.setattr(particle.progressbar, "FormatLabel", lambda text: text)
    monkeypatch.setattr(particle.progressbar, "Bar", lambda: "bar")
    monkeypatch.setattr(particle.progressbar, "AdaptiveETA", lambda: "eta")

    lab = np.zeros((3, 3, 3), dtype=np.uint32)
    lab[1, 1, 1] = 1
    lab[1, 1, 2] = 2

    volumes = particle.convexVolume(lab, nProcesses=2, verbose=True)

    assert np.array_equal(volumes, np.array([0.0, 10.0, 20.0]))


@pytest.mark.unit
def test_compactness_computes_value_for_labels_with_nonzero_centres(monkeypatch, import_with_fake_spam):
    particle = import_with_fake_spam(
        "sand_atlas.particle",
        spam_overrides={
            "getLabel": lambda lab, label, **kwargs: {"subvol": np.ones((4, 4, 4), dtype=np.uint8)},
        },
    )

    monkeypatch.setattr(particle.scipy.ndimage, "gaussian_filter", lambda array, sigma: array)
    monkeypatch.setattr(
        particle.skimage.measure,
        "marching_cubes",
        lambda volume, level: (np.array([[0.0, 0.0, 0.0]]), np.array([[0, 0, 0]]), None, None),
    )
    monkeypatch.setattr(particle.skimage.measure, "mesh_surface_area", lambda verts, faces: 12.0)

    lab = np.zeros((4, 4, 4), dtype=np.uint32)
    lab[1:3, 1:3, 1:3] = 1
    lab[0, 0, 0] = 2
    volumes = np.array([0.0, 8.0, 1.0], dtype=float)
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    compactness = particle.compactness(lab, volumes=volumes, boundingBoxes="boxes", centresOfMass=centres)

    expected = 36 * np.pi * volumes[1] ** 2 / 12.0**3
    assert compactness[1] == pytest.approx(expected)
    assert compactness[2] == 0.0


@pytest.mark.unit
def test_compactness_returns_nan_when_surface_area_is_zero(monkeypatch, import_with_fake_spam):
    particle = import_with_fake_spam(
        "sand_atlas.particle",
        spam_overrides={
            "getLabel": lambda lab, label, **kwargs: {"subvol": np.ones((4, 4, 4), dtype=np.uint8)},
        },
    )

    monkeypatch.setattr(particle.scipy.ndimage, "gaussian_filter", lambda array, sigma: array)
    monkeypatch.setattr(
        particle.skimage.measure,
        "marching_cubes",
        lambda volume, level: (np.array([[0.0, 0.0, 0.0]]), np.array([[0, 0, 0]]), None, None),
    )
    monkeypatch.setattr(particle.skimage.measure, "mesh_surface_area", lambda verts, faces: 0.0)

    lab = np.zeros((4, 4, 4), dtype=np.uint32)
    lab[1:3, 1:3, 1:3] = 1
    volumes = np.array([0.0, 8.0], dtype=float)
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    compactness = particle.compactness(lab, volumes=volumes, boundingBoxes="boxes", centresOfMass=centres)

    assert np.isnan(compactness[1])
