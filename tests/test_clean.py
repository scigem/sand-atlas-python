import numpy as np
import pytest
import scipy.ndimage


class SerialPool:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]


@pytest.mark.unit
def test_clean_subvolume_keeps_largest_component(import_with_fake_spam):
    clean = import_with_fake_spam("sand_atlas.clean")

    label_region = np.zeros((4, 4, 4), dtype=int)
    label_region[0:2, 0:2, 0:2] = 7
    label_region[3, 3, 3] = 7

    result = clean.clean_subvolume(7, label_region, slice(0, 4), slice(0, 4), slice(0, 4))

    assert result[0] == 7
    assert result[1].dtype == np.bool_
    assert result[1].sum() == 8
    assert result[1][3, 3, 3] == 0


@pytest.mark.unit
def test_parallel_clean_subvolumes_reassembles_cleaned_labels(monkeypatch, import_with_fake_spam):
    clean = import_with_fake_spam("sand_atlas.clean")

    lab = np.zeros((6, 6, 6), dtype=int)
    lab[1:3, 1:3, 1:3] = 1
    lab[1, 4, 4] = 1
    lab[3:5, 3:5, 3:5] = 2
    bounding_boxes = scipy.ndimage.find_objects(lab)

    monkeypatch.setattr(clean.multiprocessing, "Pool", SerialPool)
    monkeypatch.setattr(clean, "tqdm", lambda iterable, **kwargs: iterable)

    cleaned = clean.parallel_clean_subvolumes(
        lab,
        bounding_boxes,
        num_processors=1,
        removeEdgeLabels=False,
        makeLabelsSequential=False,
    )

    assert set(np.unique(cleaned)) == {0, 1, 2}
    assert cleaned[1, 4, 4] == 0
    assert np.count_nonzero(cleaned == 1) == 8
    assert np.count_nonzero(cleaned == 2) == 8


@pytest.mark.unit
def test_clean_labels_writes_uint16_when_labels_exceed_uint8(monkeypatch, tmp_path, import_with_fake_spam):
    clean = import_with_fake_spam("sand_atlas.clean")
    file_path = tmp_path / "labels.tif"
    file_path.write_bytes(b"placeholder")

    cleaned = np.zeros((2, 2, 2), dtype=int)
    cleaned[0, 0, 0] = 300
    recorded = {}

    monkeypatch.setattr(clean.sand_atlas.io, "load_data", lambda path: np.ones((2, 2, 2), dtype=int))
    monkeypatch.setattr(clean, "parallel_clean_subvolumes", lambda lab, bounding_boxes, num_processors: cleaned)

    def fake_imwrite(path, data):
        recorded["path"] = path
        recorded["dtype"] = data.dtype
        recorded["max"] = int(data.max())

    monkeypatch.setattr(clean.tifffile, "imwrite", fake_imwrite)

    clean.clean_labels(str(file_path), num_processors=1, verbosity=0)

    assert recorded["path"].endswith("labels_clean.tif")
    assert recorded["dtype"] == np.uint16
    assert recorded["max"] == 300
