import numpy as np
import pytest


@pytest.mark.unit
def test_gray_to_bw_uses_explicit_threshold(import_with_fake_spam):
    pipeline = import_with_fake_spam()

    data = np.array([[[0.0, 1.0], [2.0, 3.0]]])

    binary = pipeline.gray_to_bw(data, threshold=1.5)

    assert binary.dtype == np.bool_
    assert np.array_equal(binary, np.array([[[False, False], [True, True]]]))


@pytest.mark.unit
def test_label_binary_data_filters_small_regions_and_relabels(import_with_fake_spam):
    pipeline = import_with_fake_spam()

    binary = np.zeros((5, 5, 5), dtype=bool)
    binary[1:3, 1:3, 1:3] = True
    binary[4, 4, 4] = True

    labelled = pipeline.label_binary_data(binary, minimum_voxels=2)

    assert set(np.unique(labelled)) == {0, 1}
    assert np.all(labelled[1:3, 1:3, 1:3] == 1)
    assert labelled[4, 4, 4] == 0


@pytest.mark.unit
def test_remove_disconnected_regions_keeps_only_largest_component(import_with_fake_spam):
    pipeline = import_with_fake_spam()

    crop = np.zeros((5, 5, 5), dtype=bool)
    crop[0:2, 0:2, 0:2] = True
    crop[4, 4, 4] = True

    cleaned = pipeline.remove_disconnected_regions(crop)

    assert cleaned.dtype == np.bool_
    assert cleaned.sum() == 8
    assert cleaned[4, 4, 4] == 0


@pytest.mark.unit
def test_bin_data_trims_to_factor_and_preserves_dtype(import_with_fake_spam):
    pipeline = import_with_fake_spam()

    data = np.arange(5 * 6 * 7, dtype=np.uint16).reshape(5, 6, 7)

    binned = pipeline.bin_data(data, 2)

    assert binned.shape == (2, 3, 3)
    assert binned.dtype == np.uint16
    expected_first_block = np.median(data[:2, :2, :2]).astype(np.uint16)
    assert binned[0, 0, 0] == expected_first_block


@pytest.mark.unit
def test_safe_true_sphericity_reraises_unrelated_value_errors(import_with_fake_spam):
    pipeline = import_with_fake_spam(
        spam_overrides={
            "trueSphericity": lambda lab, **kwargs: (_ for _ in ()).throw(ValueError("different failure")),
        }
    )

    labelled = np.zeros((3, 3, 3), dtype=int)
    labelled[1, 1, 1] = 1

    with pytest.raises(ValueError, match="different failure"):
        pipeline.safe_true_sphericity(labelled)


@pytest.mark.unit
def test_safe_compactness_reraises_unrelated_value_errors(import_with_fake_spam):
    pipeline = import_with_fake_spam()

    def fail_compactness(lab, **kwargs):
        raise ValueError("different failure")

    pipeline.sand_atlas.particle.compactness = fail_compactness

    labelled = np.zeros((3, 3, 3), dtype=int)
    labelled[1, 1, 1] = 1

    with pytest.raises(ValueError, match="different failure"):
        pipeline.safe_compactness(labelled)
