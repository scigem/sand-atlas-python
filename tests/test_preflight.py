import json

import numpy as np
import pytest


@pytest.mark.unit
def test_rate_uses_project_threshold_convention(import_with_fake_spam):
    preflight = import_with_fake_spam("sand_atlas.preflight")

    assert preflight.rate(1.0, (3, 5))[1] == "Good"
    assert preflight.rate(4.0, (3, 5))[1] == "Moderate"
    assert preflight.rate(6.0, (3, 5))[1] == "Poor"


@pytest.mark.feature
def test_preflight_script_writes_json_summary(monkeypatch, tmp_path, import_with_fake_spam):
    preflight = import_with_fake_spam("sand_atlas.preflight")
    output_path = tmp_path / "summary.json"

    monkeypatch.setattr(
        preflight.argparse.ArgumentParser,
        "parse_args",
        lambda self: preflight.argparse.Namespace(raw="raw.tif", binning=None, output=str(output_path)),
    )
    monkeypatch.setattr(preflight.sand_atlas.io, "load_data", lambda _: np.ones((4, 4, 4), dtype=float))
    monkeypatch.setattr(
        preflight,
        "compute_all_metrics",
        lambda volume, debug=False: {"Contrast and Noise": {"Global SNR": 4.2}},
    )
    monkeypatch.setattr(preflight, "print_report", lambda metrics: None)

    preflight.preflight_script()

    assert json.loads(output_path.read_text()) == {"Contrast and Noise": {"Global SNR": 4.2}}
