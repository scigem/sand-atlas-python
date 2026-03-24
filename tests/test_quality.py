import numpy as np
import pytest

import sand_atlas.quality as quality


@pytest.mark.unit
def test_global_snr_matches_mean_over_standard_deviation(capsys):
    volume = np.arange(1, 9, dtype=float).reshape(2, 2, 2)

    snr = quality.global_snr(volume, debug=True)

    assert snr == pytest.approx(np.mean(volume) / np.std(volume))
    assert "SNR =" in capsys.readouterr().out


@pytest.mark.unit
def test_modality_index_uses_multi_otsu_thresholds(monkeypatch, capsys):
    volume = np.array([0, 1, 2, 3, 4, 5], dtype=float).reshape(1, 1, 6)

    monkeypatch.setattr(quality.filters, "threshold_multiotsu", lambda data, classes=3: np.array([2.0, 4.0]))

    result = quality.modality_index(volume, debug=True)

    assert result == pytest.approx(4.0)
    captured = capsys.readouterr().out
    assert "Thresholds = [2. 4.]" in captured
    assert "Modality Index =" in captured


@pytest.mark.unit
def test_image_entropy_returns_zero_for_uniform_volume(capsys):
    volume = np.zeros((4, 4, 4), dtype=float)

    result = quality.image_entropy(volume, window_size=2, debug=True)

    assert result == 0.0
    assert "Local Entropy Std = 0.000" in capsys.readouterr().out


@pytest.mark.unit
def test_fft_peak_frequency_reports_peak_radial_bin(monkeypatch, capsys):
    transformed = np.zeros((5, 5, 5), dtype=complex)
    transformed[2, 2, 3] = 10 + 0j

    monkeypatch.setattr(quality, "fftn", lambda volume: transformed)
    monkeypatch.setattr(quality, "fftshift", lambda volume: volume)

    result = quality.fft_peak_frequency(np.zeros((5, 5, 5), dtype=float), debug=True)

    assert result == pytest.approx(0.5)
    assert "FFT Peak Frequency = 0.5" in capsys.readouterr().out


@pytest.mark.unit
def test_autocorrelation_range_returns_first_index_below_threshold(monkeypatch, capsys):
    corr = np.zeros((3, 3, 5), dtype=float)
    corr[1, 1, :] = np.array([0.6, 0.8, 1.0, 0.4, 0.2])

    monkeypatch.setattr(quality, "fftconvolve", lambda left, right, mode: corr)

    result = quality.autocorrelation_range(np.ones((3, 3, 5), dtype=float), debug=True)

    assert result == 3
    assert "Autocorrelation Range = 3" in capsys.readouterr().out


@pytest.mark.unit
def test_edge_density_finds_edges_in_step_volume(capsys):
    volume = np.zeros((7, 7, 7), dtype=float)
    volume[2:5, 2:5, 2:5] = 1.0

    density = quality.edge_density(volume, debug=True)

    assert 0 < density < 1
    assert "Edge Density =" in capsys.readouterr().out


@pytest.mark.unit
def test_fractal_dimension_uses_box_counts(monkeypatch, capsys):
    recorded = {}

    def fake_polyfit(x_values, y_values, degree):
        recorded["x"] = x_values
        recorded["y"] = y_values
        recorded["degree"] = degree
        return np.array([-1.7, 0.0])

    monkeypatch.setattr(quality.np, "polyfit", fake_polyfit)

    result = quality.fractal_dimension(np.ones((16, 16, 16), dtype=float), debug=True)

    assert result == pytest.approx(1.7)
    assert recorded["degree"] == 1
    assert len(recorded["x"]) == 2
    assert len(recorded["y"]) == 2
    assert "Fractal Dimension = 1.7" in capsys.readouterr().out


@pytest.mark.unit
def test_gradient_std_returns_high_inverted_score_for_flat_volume(capsys):
    result = quality.gradient_std(np.zeros((4, 4, 4), dtype=float), debug=True)

    assert result == pytest.approx(10.0)
    captured = capsys.readouterr().out
    assert "Gradient Std = 0.000" in captured
    assert "Inverted Metric (lower=better) = 10.000" in captured


@pytest.mark.unit
def test_otsu_solid_fraction_ignores_mask_value(monkeypatch, capsys):
    seen = {}

    def fake_threshold_otsu(data):
        seen["data"] = data.copy()
        return 1.0

    volume = np.array([0, 0, 1, 2], dtype=float).reshape(1, 2, 2)
    monkeypatch.setattr(quality.filters, "threshold_otsu", fake_threshold_otsu)

    result = quality.otsu_solid_fraction(volume, debug=True, maskValue=0)

    assert np.array_equal(seen["data"], np.array([1.0, 2.0]))
    assert result == pytest.approx(0.25)
    assert "Otsu Solid Fraction = 0.25" in capsys.readouterr().out


@pytest.mark.unit
def test_local_solid_fraction_std_aggregates_subvolume_measurements(monkeypatch, capsys):
    volume = np.zeros((4, 4, 4), dtype=float)
    volume[:2, :2, :2] = 1.0
    volume[2:, 2:, 2:] = 1.0

    monkeypatch.setattr(quality, "otsu_solid_fraction", lambda subvol: float(np.mean(subvol)))

    result = quality.local_solid_fraction_std(volume, window_size=2, debug=True)

    assert result > 0
    assert "Local Solid Fraction Std =" in capsys.readouterr().out
