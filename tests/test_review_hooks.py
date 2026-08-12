from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-review-hooks.sh"
REVIEWER = ROOT / "scripts" / "janki-review.sh"


def run(*command: str | Path, cwd: Path, env: dict[str, str] | None = None):
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def git_repository(path: Path) -> None:
    assert run("git", "init", "-q", cwd=path).returncode == 0
    assert run("git", "config", "user.name", "Hook Test", cwd=path).returncode == 0
    assert run("git", "config", "user.email", "hook@example.invalid", cwd=path).returncode == 0


def installer_env(root: Path) -> dict[str, str]:
    return {**os.environ, "JANKI_REPO_ROOT": str(root)}


def test_installer_puts_both_tracked_shims_in_a_fresh_clone(tmp_path: Path) -> None:
    git_repository(tmp_path)

    result = run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path))

    assert result.returncode == 0, result.stderr
    for name in ("post-commit", "pre-push"):
        installed = tmp_path / ".git" / "hooks" / name
        assert installed.read_bytes() == (ROOT / "scripts" / "git-hooks" / name).read_bytes()
        assert installed.stat().st_mode & stat.S_IXUSR
        assert "scripts/janki-review.sh" in installed.read_text()


def test_installer_refuses_all_changes_when_one_hook_is_unmanaged(tmp_path: Path) -> None:
    git_repository(tmp_path)
    hooks = tmp_path / ".git" / "hooks"
    existing = hooks / "pre-push"
    existing.write_text("#!/bin/sh\necho mine\n")
    existing.chmod(0o755)

    result = run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path))

    assert result.returncode == 1
    assert existing.read_text() == "#!/bin/sh\necho mine\n"
    assert not (hooks / "post-commit").exists(), "installation is all-or-nothing"
    assert "unmanaged hook" in result.stderr


def review_repository(path: Path) -> Path:
    git_repository(path)
    scripts = path / "scripts"
    scripts.mkdir()
    reviewer = scripts / "janki-review.sh"
    shutil.copy2(REVIEWER, reviewer)

    tracked = path / "change.txt"
    tracked.write_text("first\n")
    assert run("git", "add", "change.txt", cwd=path).returncode == 0
    assert run("git", "commit", "-qm", "first", cwd=path).returncode == 0
    tracked.write_text("second\n")
    assert run("git", "commit", "-qam", "second", cwd=path).returncode == 0
    return reviewer


def test_disabled_marker_is_announced_before_any_review_runs(tmp_path: Path) -> None:
    reviewer = review_repository(tmp_path)
    result = run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path))
    assert result.returncode == 0, result.stderr
    marker = tmp_path / ".claude" / "hooks" / "DISABLED"
    marker.parent.mkdir(parents=True)
    marker.write_text("budget pause\n")

    result = run(reviewer, "gate", "not-a-range", "disabled", cwd=tmp_path)

    assert result.returncode == 0
    assert "janki review: disabled (budget pause)" in result.stdout
    assert "rm .claude/hooks/DISABLED" in result.stdout
    assert not (tmp_path / ".claude" / "reviews").exists()

    post_commit = run(tmp_path / ".git" / "hooks" / "post-commit", cwd=tmp_path)
    assert post_commit.returncode == 0
    assert "janki review: disabled (budget pause)" in post_commit.stdout
    assert "started in background" not in post_commit.stdout


def test_failed_reviewer_writes_an_error_verdict_and_allows_push(tmp_path: Path) -> None:
    reviewer = review_repository(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    claude = fake_bin / "claude"
    claude.write_text("#!/bin/sh\nexit 7\n")
    claude.chmod(0o755)
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    result = run(reviewer, "gate", "HEAD~1..HEAD", "failed", cwd=tmp_path, env=env)

    assert result.returncode == 0
    assert "NOTHING WAS REVIEWED" in result.stdout
    report = (tmp_path / ".claude" / "reviews" / "failed.md").read_text()
    assert "_Review did not complete: the reviewer exited 7._" in report
    assert report.rstrip().endswith("VERDICT: ERROR")
