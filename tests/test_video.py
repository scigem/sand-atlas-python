import subprocess

import pytest

import sand_atlas.video


@pytest.mark.unit
def test_resolve_path_for_blender_returns_local_path_outside_wsl(monkeypatch):
    monkeypatch.setattr(sand_atlas.video.platform, "uname", lambda: type("U", (), {"release": "Darwin"})())

    resolved = sand_atlas.video.resolve_path_for_blender("blender_scripts/render_mesh.py")

    assert resolved.endswith("sand_atlas/blender_scripts/render_mesh.py")


@pytest.mark.unit
def test_resolve_path_for_blender_uses_wslpath_when_running_under_wsl(monkeypatch):
    monkeypatch.setattr(sand_atlas.video.platform, "uname", lambda: type("U", (), {"release": "microsoft-standard"})())
    monkeypatch.setattr(
        sand_atlas.video.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, stdout="C:\\render_mesh.py\n"),
    )

    resolved = sand_atlas.video.resolve_path_for_blender("blender_scripts/render_mesh.py")

    assert resolved == "C:\\render_mesh.py"


@pytest.mark.feature
def test_make_website_video_writes_sources_and_runs_grid_pipeline(monkeypatch, tmp_path):
    stl_dir = tmp_path / "stl_ORIGINAL"
    output_dir = tmp_path / "upload"
    stl_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    for index in range(2):
        (stl_dir / f"particle_{index:05}.stl").write_text("solid")
        (stl_dir / f"particle_{index:05}.webm").write_text("video")
    (stl_dir / "blank.webm").write_text("blank")

    commands = []

    monkeypatch.setattr(sand_atlas.video, "resolve_path_for_blender", lambda path: f"/abs/{path}")
    monkeypatch.setattr(sand_atlas.video, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(sand_atlas.video.os, "system", lambda command: commands.append(command) or 0)

    sand_atlas.video.make_website_video(str(stl_dir), str(output_dir), debug=False)

    assert (tmp_path / "sources.txt").read_text() == "file grid_0.webm\n"
    assert any("xstack=inputs=12" in command for command in commands)
    assert any("-f concat -safe 0 -i sources.txt -c copy all_particles.webm" in command for command in commands)
    assert any(f"{output_dir}/all_particles.webm" in command for command in commands)
    assert any(command == "rm grid_*.webm" for command in commands)


@pytest.mark.feature
def test_make_individual_videos_builds_render_and_encode_commands(monkeypatch, tmp_path):
    stl_dir = tmp_path / "stl_ORIGINAL"
    output_dir = tmp_path / "media"
    stl_dir.mkdir()
    (stl_dir / "particle_00000.stl").write_text("solid")

    commands = []

    monkeypatch.setattr(sand_atlas.video, "resolve_path_for_blender", lambda path: f"/abs/{path}")
    monkeypatch.setattr(sand_atlas.video, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(sand_atlas.video.os, "system", lambda command: commands.append(command) or 0)

    sand_atlas.video.make_individual_videos(
        str(stl_dir),
        str(output_dir),
        max_videos=1,
        bg_colour="#ffffff",
        fg_colour="#000000",
        debug=True,
    )

    assert output_dir.exists()
    assert any(
        "blender --background -t 4 -noaudio --python /abs/blender_scripts/render_mesh.py" in command
        for command in commands
    )
    assert any("--bg_colour #ffffff --fg_colour #000000 --resolution 1080x1350" in command for command in commands)
    assert any("ffmpeg -y -framerate 30 -pattern_type glob" in command for command in commands)
