import os
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "calculate_all_particle_properties.sh"


def _run_script(tmp_path, json_dir=None, sands_dir=None, path_prefix=None):
    env = os.environ.copy()
    env["JSON_DIR"] = str(json_dir if json_dir is not None else tmp_path / "missing-json")
    env["SANDS_DIR"] = str(sands_dir if sands_dir is not None else tmp_path / "missing-sands")
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}{os.pathsep}{env['PATH']}"
    return subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)


@pytest.mark.feature
def test_script_fails_when_json_directory_missing(tmp_path):
    result = _run_script(tmp_path)

    assert result.returncode == 1
    assert "JSON directory not found" in result.stderr


@pytest.mark.feature
def test_script_fails_when_sands_directory_missing(tmp_path):
    json_dir = tmp_path / "json"
    json_dir.mkdir()

    result = _run_script(tmp_path, json_dir=json_dir)

    assert result.returncode == 1
    assert "Sands directory not found" in result.stderr


@pytest.mark.feature
def test_script_reports_no_matches_when_pairs_are_missing(tmp_path):
    json_dir = tmp_path / "json"
    sands_dir = tmp_path / "sands"
    json_dir.mkdir()
    sands_dir.mkdir()
    (json_dir / "sample.json").write_text("{}")

    result = _run_script(tmp_path, json_dir=json_dir, sands_dir=sands_dir)

    assert result.returncode == 1
    assert "No matching JSON/labelled TIFF pairs found" in result.stderr
    assert "sample folder does not exist" in result.stderr


@pytest.mark.feature
def test_script_processes_matching_pairs_and_calls_properties(tmp_path):
    json_dir = tmp_path / "json"
    sands_dir = tmp_path / "sands"
    bin_dir = tmp_path / "bin"
    json_dir.mkdir()
    sands_dir.mkdir()
    bin_dir.mkdir()

    sample_dir = sands_dir / "sample"
    sample_dir.mkdir()
    (json_dir / "sample.json").write_text('{"microns_per_pixel": "1.0"}')
    (sample_dir / "labelled.tif").write_bytes(b"fake-tif")

    call_log = tmp_path / "calls.txt"
    fake_properties = bin_dir / "sand_atlas_properties"
    fake_properties.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\n\' "$*" >> "$CALL_LOG"\n'
        'output=""\n'
        "while [[ $# -gt 0 ]]; do\n"
        '  if [[ "$1" == "--output" ]]; then\n'
        '    output="$2"\n'
        "    shift 2\n"
        "    continue\n"
        "  fi\n"
        "  shift\n"
        "done\n"
        "printf 'Particle ID,Volume\\n1,10\\n' > \"$output\"\n"
    )
    fake_properties.chmod(0o755)

    env = os.environ.copy()
    env["CALL_LOG"] = str(call_log)
    env["JSON_DIR"] = str(json_dir)
    env["SANDS_DIR"] = str(sands_dir)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(["bash", str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert result.returncode == 0
    assert "Processing sample" in result.stdout
    assert call_log.read_text().strip() == (
        f"{json_dir / 'sample.json'} {sample_dir / 'labelled.tif'} --output {sample_dir / 'summary.csv'}"
    )
    assert (sample_dir / "summary.csv").exists()
