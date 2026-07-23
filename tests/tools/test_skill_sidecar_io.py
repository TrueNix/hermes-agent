from pathlib import Path

import pytest

from tools import skill_sidecar_io as sidecar


@pytest.fixture
def path_backend(monkeypatch):
    monkeypatch.setattr(sidecar, "_DIR_FD_SUPPORTED", False)
    monkeypatch.setattr(sidecar, "_WINDOWS_PATH_BACKEND", True)


def test_path_backend_round_trips_atomic_append_and_read(
    tmp_path: Path, path_backend
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()

    sidecar.atomic_write_sidecar(skill_dir, ".state.json", b"first")
    sidecar.atomic_write_sidecar(skill_dir, ".state.json", b"second")
    sidecar.append_sidecar(
        skill_dir,
        ".memory.md",
        b"entry\n",
        lock_name=".memory.lock",
        dedupe_marker=b"entry",
    )
    sidecar.append_sidecar(
        skill_dir,
        ".memory.md",
        b"entry\n",
        lock_name=".memory.lock",
        dedupe_marker=b"entry",
    )

    state, truncated = sidecar.read_sidecar(
        skill_dir, ".state.json", max_bytes=100
    )
    memory, _ = sidecar.read_sidecar(skill_dir, ".memory.md", max_bytes=100)
    assert state == b"second"
    assert truncated is False
    assert memory == b"entry\n"


def test_path_backend_rejects_sidecar_symlink(tmp_path: Path, path_backend) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    link = skill_dir / ".state.json"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(OSError, match="symlink"):
        sidecar.read_sidecar(skill_dir, ".state.json", max_bytes=100)


def test_platform_without_secure_backend_still_fails_closed(
    tmp_path: Path, monkeypatch
) -> None:
    skill_dir = tmp_path / "skill"
    skill_dir.mkdir()
    monkeypatch.setattr(sidecar, "_DIR_FD_SUPPORTED", False)
    monkeypatch.setattr(sidecar, "_WINDOWS_PATH_BACKEND", False)

    assert sidecar.secure_sidecar_io_available() is False
    with pytest.raises(OSError, match="unavailable"):
        sidecar.atomic_write_sidecar(skill_dir, ".state.json", b"data")
