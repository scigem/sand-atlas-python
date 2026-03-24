import numpy as np


import pytest


@pytest.mark.unit
def test_filter_particles_touching_edges_relabels_remaining_particles(import_with_fake_spam):
    pipeline = import_with_fake_spam()

    labelled_data = np.zeros((5, 5, 5), dtype=int)
    labelled_data[0, 2, 2] = 10
    labelled_data[1:4, 1:4, 1:4] = 20

    filtered = pipeline.filter_particles_touching_edges(labelled_data)

    assert set(np.unique(filtered)) == {0, 1}
    assert np.all(filtered[1:4, 1:4, 1:4] == 1)


@pytest.mark.unit
def test_get_particle_properties_excludes_edge_particles_from_summary(monkeypatch, import_with_fake_spam):
    pipeline = import_with_fake_spam()
    recorded = {}

    def fake_volumes(lab, **kwargs):
        recorded["unique_labels"] = np.unique(lab).tolist()
        return np.array([0, np.count_nonzero(lab == 1)], dtype=float)

    monkeypatch.setattr(pipeline.spam.label, "volumes", fake_volumes)
    monkeypatch.setattr(pipeline.spam.label, "boundingBoxes", lambda lab: None)
    monkeypatch.setattr(pipeline.spam.label, "centresOfMass", lambda lab, **kwargs: np.zeros((2, 3)))
    monkeypatch.setattr(pipeline.spam.label, "equivalentRadii", lambda lab, **kwargs: np.array([0, 1], dtype=float))
    monkeypatch.setattr(
        pipeline.spam.label, "ellipseAxes", lambda lab, volumes: np.array([[0, 0, 0], [3, 2, 1]], dtype=float)
    )
    monkeypatch.setattr(pipeline.spam.label, "trueSphericity", lambda lab, **kwargs: np.array([0, 0.8], dtype=float))
    monkeypatch.setattr(pipeline.spam.label, "convexVolume", lambda lab, **kwargs: np.array([0, 10], dtype=float))
    monkeypatch.setattr(
        pipeline.sand_atlas.particle, "compactness", lambda lab, **kwargs: np.array([0, 0.7], dtype=float)
    )

    labelled_data = np.zeros((5, 5, 5), dtype=int)
    labelled_data[0, 2, 2] = 1
    labelled_data[1:4, 1:4, 1:4] = 2

    df = pipeline.get_particle_properties(labelled_data, microns_per_voxel=1.0)

    assert recorded["unique_labels"] == [0, 1]
    assert list(df.index) == [1]
    assert df.shape[0] == 1


@pytest.mark.unit
def test_get_particle_properties_returns_empty_dataframe_when_all_particles_touch_edges(import_with_fake_spam):
    pipeline = import_with_fake_spam()

    labelled_data = np.zeros((4, 4, 4), dtype=int)
    labelled_data[0, 1:3, 1:3] = 1

    df = pipeline.get_particle_properties(labelled_data, microns_per_voxel=1.0)

    assert df.empty
    assert list(df.columns) == [
        "Volume (µm³)",
        "Equivalent Diameter (µm)",
        "Major Axis Length (µm)",
        "Middle Axis Length (µm)",
        "Minor Axis Length (µm)",
        "True Sphericity (-)",
        "Convexity (-)",
        "Flatness (-)",
        "Elongation (-)",
        "Compactness (-)",
    ]


@pytest.mark.unit
def test_get_particle_properties_handles_degenerate_sphericity_particles(monkeypatch, import_with_fake_spam):
    pipeline = import_with_fake_spam()

    monkeypatch.setattr(pipeline.spam.label, "boundingBoxes", lambda lab: None)
    monkeypatch.setattr(pipeline.spam.label, "centresOfMass", lambda lab, **kwargs: np.zeros((int(lab.max()) + 1, 3)))
    monkeypatch.setattr(pipeline.spam.label, "volumes", lambda lab, **kwargs: np.array([0, 8, 27], dtype=float))
    monkeypatch.setattr(pipeline.spam.label, "equivalentRadii", lambda lab, **kwargs: np.array([0, 1, 2], dtype=float))
    monkeypatch.setattr(
        pipeline.spam.label,
        "ellipseAxes",
        lambda lab, volumes: np.array([[0, 0, 0], [3, 2, 1], [4, 3, 2]], dtype=float),
    )

    def fake_true_sphericity(lab, **kwargs):
        max_label = int(lab.max())
        if max_label > 1:
            raise ValueError("Surface level must be within volume data range.")
        if np.count_nonzero(lab == 1) == 8:
            return np.array([0, 0.75], dtype=float)
        raise ValueError("Surface level must be within volume data range.")

    monkeypatch.setattr(pipeline.spam.label, "trueSphericity", fake_true_sphericity)
    monkeypatch.setattr(pipeline.spam.label, "convexVolume", lambda lab, **kwargs: np.array([0, 10, 30], dtype=float))
    monkeypatch.setattr(
        pipeline.sand_atlas.particle, "compactness", lambda lab, **kwargs: np.array([0, 0.7, 0.8], dtype=float)
    )

    labelled_data = np.zeros((8, 8, 8), dtype=int)
    labelled_data[1:3, 1:3, 1:3] = 1
    labelled_data[4:7, 4:7, 4:7] = 2

    df = pipeline.get_particle_properties(labelled_data, microns_per_voxel=1.0)

    assert df.loc[1, "True Sphericity (-)"] == 0.75
    assert np.isnan(df.loc[2, "True Sphericity (-)"])


@pytest.mark.unit
def test_get_particle_properties_handles_degenerate_compactness_particles(monkeypatch, import_with_fake_spam):
    pipeline = import_with_fake_spam()

    monkeypatch.setattr(pipeline.spam.label, "boundingBoxes", lambda lab: None)
    monkeypatch.setattr(pipeline.spam.label, "centresOfMass", lambda lab, **kwargs: np.zeros((int(lab.max()) + 1, 3)))
    monkeypatch.setattr(pipeline.spam.label, "volumes", lambda lab, **kwargs: np.array([0, 8, 27], dtype=float))
    monkeypatch.setattr(pipeline.spam.label, "equivalentRadii", lambda lab, **kwargs: np.array([0, 1, 2], dtype=float))
    monkeypatch.setattr(
        pipeline.spam.label,
        "ellipseAxes",
        lambda lab, volumes: np.array([[0, 0, 0], [3, 2, 1], [4, 3, 2]], dtype=float),
    )
    monkeypatch.setattr(pipeline.spam.label, "trueSphericity", lambda lab, **kwargs: np.array([0, 0.75, 0.85], dtype=float))
    monkeypatch.setattr(pipeline.spam.label, "convexVolume", lambda lab, **kwargs: np.array([0, 10, 30], dtype=float))

    def fake_compactness(lab, **kwargs):
        max_label = int(lab.max())
        if max_label > 1:
            raise ValueError("Surface level must be within volume data range.")
        if np.count_nonzero(lab == 1) == 8:
            return np.array([0, 0.7], dtype=float)
        raise ValueError("Surface level must be within volume data range.")

    monkeypatch.setattr(pipeline.sand_atlas.particle, "compactness", fake_compactness)

    labelled_data = np.zeros((8, 8, 8), dtype=int)
    labelled_data[1:3, 1:3, 1:3] = 1
    labelled_data[4:7, 4:7, 4:7] = 2

    df = pipeline.get_particle_properties(labelled_data, microns_per_voxel=1.0)

    assert df.loc[1, "Compactness (-)"] == 0.7
    assert np.isnan(df.loc[2, "Compactness (-)"])
