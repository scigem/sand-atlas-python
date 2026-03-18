import importlib
import sys
import types

import numpy as np


def import_pipeline_with_fake_spam():
    fake_spam = types.ModuleType("spam")
    fake_spam_label = types.ModuleType("spam.label")

    fake_spam_label.boundingBoxes = lambda lab: None
    fake_spam_label.centresOfMass = lambda lab, **kwargs: np.zeros((int(lab.max()) + 1, 3))
    fake_spam_label.volumes = lambda lab, **kwargs: np.arange(int(lab.max()) + 1)
    fake_spam_label.equivalentRadii = lambda lab, **kwargs: np.arange(int(lab.max()) + 1)
    fake_spam_label.ellipseAxes = lambda lab, volumes: np.ones((int(lab.max()) + 1, 3))
    fake_spam_label.trueSphericity = lambda lab, **kwargs: np.ones(int(lab.max()) + 1)
    fake_spam_label.convexVolume = lambda lab, **kwargs: np.arange(int(lab.max()) + 1) + 1
    fake_spam_label.makeLabelsSequential = lambda lab: lab
    fake_spam_label.label = types.SimpleNamespace(makeLabelsSequential=lambda lab: lab)

    fake_spam.label = fake_spam_label

    sys.modules["spam"] = fake_spam
    sys.modules["spam.label"] = fake_spam_label

    for module_name in ["sand_atlas.pipeline", "sand_atlas.particle", "sand_atlas.clean"]:
        sys.modules.pop(module_name, None)

    return importlib.import_module("sand_atlas.pipeline")


def test_filter_particles_touching_edges_relabels_remaining_particles():
    pipeline = import_pipeline_with_fake_spam()

    labelled_data = np.zeros((5, 5, 5), dtype=int)
    labelled_data[0, 2, 2] = 10
    labelled_data[1:4, 1:4, 1:4] = 20

    filtered = pipeline.filter_particles_touching_edges(labelled_data)

    assert set(np.unique(filtered)) == {0, 1}
    assert np.all(filtered[1:4, 1:4, 1:4] == 1)


def test_get_particle_properties_excludes_edge_particles_from_summary(monkeypatch):
    pipeline = import_pipeline_with_fake_spam()
    recorded = {}

    def fake_volumes(lab, **kwargs):
        recorded["unique_labels"] = np.unique(lab).tolist()
        return np.array([0, 8], dtype=float)

    monkeypatch.setattr(pipeline.spam.label, "volumes", fake_volumes)
    monkeypatch.setattr(pipeline.spam.label, "boundingBoxes", lambda lab: None)
    monkeypatch.setattr(pipeline.spam.label, "centresOfMass", lambda lab, **kwargs: np.zeros((2, 3)))
    monkeypatch.setattr(pipeline.spam.label, "equivalentRadii", lambda lab, **kwargs: np.array([0, 1], dtype=float))
    monkeypatch.setattr(pipeline.spam.label, "ellipseAxes", lambda lab, volumes: np.array([[0, 0, 0], [3, 2, 1]], dtype=float))
    monkeypatch.setattr(pipeline.spam.label, "trueSphericity", lambda lab, **kwargs: np.array([0, 0.8], dtype=float))
    monkeypatch.setattr(pipeline.spam.label, "convexVolume", lambda lab, **kwargs: np.array([0, 10], dtype=float))
    monkeypatch.setattr(pipeline.sand_atlas.particle, "compactness", lambda lab, **kwargs: np.array([0, 0.7], dtype=float))

    labelled_data = np.zeros((5, 5, 5), dtype=int)
    labelled_data[0, 2, 2] = 1
    labelled_data[1:4, 1:4, 1:4] = 2

    df = pipeline.get_particle_properties(labelled_data, microns_per_voxel=1.0)

    assert recorded["unique_labels"] == [0, 1]
    assert list(df.index) == [1]
    assert df.shape[0] == 1
