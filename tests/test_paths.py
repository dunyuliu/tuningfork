"""Paths as a person types them: translated for WSL, resolved against a named base."""

from __future__ import annotations

from pathlib import Path

from tuningfork.paths import host_path, resolve_path, windows_form, windows_path

WSL = {"wsl": True, "distro": "Ubuntu-24.04", "mounts": "/mnt/"}


def test_a_drive_path_is_where_wsl_mounts_it():
    assert host_path(r"C:\Users\me\skills\tally", **WSL) == "/mnt/c/Users/me/skills/tally"
    assert host_path("D:/data/evals.json", **WSL) == "/mnt/d/data/evals.json"
    assert host_path("c:\\", **WSL) == "/mnt/c"


def test_explorer_quotes_and_spaces_go():
    assert host_path('  "C:\\Users\\me\\My Skills\\tally"  ', **WSL) == "/mnt/c/Users/me/My Skills/tally"
    assert host_path("'/home/me/skill'", **WSL) == "/home/me/skill"


def test_a_moved_mount_root_is_honoured():
    assert host_path(r"C:\x", wsl=True, distro="", mounts="/windir/") == "/windir/c/x"


def test_this_distros_share_is_a_path_inside_it():
    assert host_path(r"\\wsl.localhost\Ubuntu-24.04\home\me\skill", **WSL) == "/home/me/skill"
    assert host_path(r"\\wsl$\ubuntu-24.04\home\me", **WSL) == "/home/me"


def test_another_distros_share_is_left_alone():
    """It names files this distro cannot open by that path: left as given, it is reported
    missing rather than quietly turned into something else."""
    other = r"\\wsl.localhost\Ubuntu\home\me\skill"
    assert host_path(other, **WSL) == other


def test_backslashes_in_a_relative_path_become_slashes_under_wsl():
    assert host_path(r"skills\tally", **WSL) == "skills/tally"


def test_outside_wsl_nothing_is_translated():
    assert host_path(r"C:\Users\me", wsl=False) == r"C:\Users\me"
    assert host_path(r"skills\tally", wsl=False) == r"skills\tally"


def test_a_home_folder_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert host_path("~/skills", wsl=False) == f"{tmp_path}/skills"


def test_windows_paths_are_recognised():
    assert windows_path(r"C:\Users\me\claude.exe")
    assert windows_path('"D:/tools/hermes"')
    assert windows_path(r"\\server\share\x")
    assert not windows_path("/home/me/.local/bin/hermes")


def test_a_relative_path_says_what_it_was_resolved_against(tmp_path):
    (tmp_path / "skills" / "tally").mkdir(parents=True)
    found = resolve_path("skills/tally", tmp_path, wsl=False)
    assert found.path == tmp_path / "skills" / "tally"
    assert found.base == tmp_path
    assert found.exists
    assert found.translated_from is None
    assert found.to_json() == {
        "given": "skills/tally",
        "path": str(tmp_path / "skills" / "tally"),
        "relative_to": str(tmp_path),
        "translated_from": None,
        "exists": True,
    }


def test_an_absolute_path_has_no_base(tmp_path):
    found = resolve_path(str(tmp_path / "missing"), Path("/elsewhere"), wsl=False)
    assert found.base is None
    assert not found.exists


def test_a_translated_path_says_what_it_was_given():
    found = resolve_path(r"C:\Users\me\skill", Path("/base"), **WSL)
    assert found.path == Path("/mnt/c/Users/me/skill")
    assert found.translated_from == r"C:\Users\me\skill"
    assert found.base is None


def test_a_path_is_named_as_windows_would_name_it():
    """What a person is shown before a file is written: where Explorer finds it."""
    assert windows_form("/mnt/c/Users/me/skills/SKILL.md", wsl=True, mounts="/mnt/") == r"C:\Users\me\skills\SKILL.md"
    assert windows_form("/mnt/d", wsl=True, mounts="/mnt/") == "D:\\"
    assert windows_form("/home/me/s/SKILL.md", wsl=True, distro="Ubuntu-24.04", mounts="/mnt/") == \
        r"\\wsl.localhost\Ubuntu-24.04\home\me\s\SKILL.md"
    assert windows_form("/home/me", wsl=True, distro="", mounts="/mnt/") is None
    assert windows_form("/mnt/c/x", wsl=False) is None


def test_the_windows_form_is_the_reverse_of_host_path():
    for path in ("/mnt/c/Users/me/skills/SKILL.md", "/home/me/skills/SKILL.md"):
        shown = windows_form(path, wsl=True, distro="Ubuntu-24.04", mounts="/mnt/")
        assert host_path(shown, wsl=True, distro="Ubuntu-24.04", mounts="/mnt/") == path
