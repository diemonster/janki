"""The pre-push hook must abort rather than dispatch Git LFS with refs git
never sent, so a failed capture of git's ref list stops the push."""

import os
import stat
import subprocess
from pathlib import Path

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "git-hooks" / "pre-push"
ZERO = "0" * 40
SHA = "1" * 40


def _fake_exe(path: Path, script: str) -> None:
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def test_failed_ref_capture_aborts_before_lfs_dispatch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "git-lfs-dispatched"
    # cat dies mid-capture, after emitting a truncated ref line to $REFS.
    _fake_exe(bin_dir / "cat", '#!/bin/sh\nprintf "refs/heads/ma"\nexit 7\n')
    # `git lfs ...` dispatches to git-lfs on PATH; the marker records the upload.
    _fake_exe(bin_dir / "git-lfs", f'#!/bin/sh\n: >"{marker}"\nexit 0\n')

    env = dict(
        os.environ,
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        HOME=str(tmp_path),
    )
    refs = (
        f"refs/heads/main {SHA} refs/heads/main {ZERO}\n"
        f"refs/heads/topic {SHA} refs/heads/topic {ZERO}\n"
    )
    result = subprocess.run(
        ["bash", str(HOOK), "origin", "https://example.invalid/janki.git"],
        cwd=repo,
        env=env,
        input=refs,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert not marker.exists(), (
        "git-lfs was dispatched even though the ref capture failed; "
        "the hook must stop before any upload"
    )
    assert result.returncode != 0, (
        f"hook exited 0 after a failed ref capture (stderr: {result.stderr!r})"
    )
