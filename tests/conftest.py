import importlib
import sys
import types

import numpy as np
import pytest


def _make_labels_sequential(lab):
    relabelled = np.zeros_like(lab)
    for new_label, label in enumerate(sorted(label for label in np.unique(lab) if label != 0), start=1):
        relabelled[lab == label] = new_label
    return relabelled


def _labels_on_edges(lab):
    edge_labels = np.unique(
        np.concatenate(
            [
                lab[0, :, :].ravel(),
                lab[-1, :, :].ravel(),
                lab[:, 0, :].ravel(),
                lab[:, -1, :].ravel(),
                lab[:, :, 0].ravel(),
                lab[:, :, -1].ravel(),
            ]
        )
    )
    return edge_labels[edge_labels != 0]


def _remove_labels(lab, labels):
    cleaned = lab.copy()
    for label in labels:
        cleaned[cleaned == label] = 0
    return cleaned


@pytest.fixture
def import_with_fake_spam():
    def _import(module_name="sand_atlas.pipeline", spam_overrides=None):
        fake_spam = types.ModuleType("spam")
        fake_spam_label = types.ModuleType("spam.label")

        defaults = {
            "boundingBoxes": lambda lab: None,
            "centresOfMass": lambda lab, **kwargs: np.zeros((int(lab.max()) + 1, 3), dtype=float),
            "volumes": lambda lab, **kwargs: np.arange(int(lab.max()) + 1, dtype=float),
            "equivalentRadii": lambda lab, **kwargs: np.arange(int(lab.max()) + 1, dtype=float),
            "ellipseAxes": lambda lab, volumes: np.ones((int(lab.max()) + 1, 3), dtype=float),
            "trueSphericity": lambda lab, **kwargs: np.ones(int(lab.max()) + 1, dtype=float),
            "convexVolume": lambda lab, **kwargs: np.arange(int(lab.max()) + 1, dtype=float) + 1,
            "labelsOnEdges": _labels_on_edges,
            "removeLabels": _remove_labels,
            "makeLabelsSequential": _make_labels_sequential,
        }

        if spam_overrides:
            defaults.update(spam_overrides)

        for name, value in defaults.items():
            setattr(fake_spam_label, name, value)

        fake_spam_label.label = types.SimpleNamespace(makeLabelsSequential=_make_labels_sequential)
        fake_spam.label = fake_spam_label

        sys.modules["spam"] = fake_spam
        sys.modules["spam.label"] = fake_spam_label

        for candidate in [
            "sand_atlas.pipeline",
            "sand_atlas.particle",
            "sand_atlas.clean",
            "sand_atlas.preflight",
            module_name,
        ]:
            sys.modules.pop(candidate, None)

        return importlib.import_module(module_name)

    return _import


@pytest.fixture
def sample_metadata_json(tmp_path):
    json_path = tmp_path / "sample.json"
    json_path.write_text('{"microns_per_pixel": "2.5", "URI": "sample"}')
    return json_path


@pytest.fixture
def labelled_volume():
    labelled = np.zeros((6, 6, 6), dtype=int)
    labelled[1:3, 1:3, 1:3] = 1
    labelled[3:5, 3:5, 3:5] = 2
    return labelled
