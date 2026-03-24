from pathlib import Path

import pytest

import sand_atlas.data


class DummyResponse:
    def __init__(self, json_payload=None, chunks=None):
        self._json_payload = json_payload
        self._chunks = chunks or []

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_payload

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


@pytest.mark.network
def test_list_returns_json_payload(monkeypatch):
    monkeypatch.setattr(
        sand_atlas.data.requests,
        "get",
        lambda url: DummyResponse(json_payload=["sand-a", "sand-b"]),
    )

    assert sand_atlas.data.list() == ["sand-a", "sand-b"]


@pytest.mark.network
def test_get_all_rejects_invalid_quality():
    with pytest.raises(ValueError, match="Quality must be one of"):
        sand_atlas.data.get_all("sample", "BAD")


@pytest.mark.network
def test_get_all_streams_zip_to_disk(monkeypatch, tmp_path):
    recorded = {}

    def fake_get(url, params):
        recorded["url"] = url
        recorded["params"] = params
        return DummyResponse(chunks=[b"abc", b"123"])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sand_atlas.data.requests, "get", fake_get)

    sand_atlas.data.get_all("sample", "100")

    assert recorded["params"] == {"sand": "sample", "quality": "100"}
    assert (tmp_path / "sample_100.zip").read_bytes() == b"abc123"


@pytest.mark.network
def test_get_by_id_streams_stl_to_disk(monkeypatch, tmp_path):
    recorded = {}

    def fake_get(url, params):
        recorded["url"] = url
        recorded["params"] = params
        return DummyResponse(chunks=[b"solid mesh"])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sand_atlas.data.requests, "get", fake_get)

    sand_atlas.data.get_by_id("sample", "30", 7)

    assert recorded["params"] == {"sand": "sample", "quality": "30", "id": 7}
    assert (tmp_path / "sample_30_7.stl").read_bytes() == b"solid mesh"
