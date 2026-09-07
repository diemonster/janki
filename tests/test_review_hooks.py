from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-review-hooks.sh"
REVIEWER = ROOT / "scripts" / "janki-review.sh"
CODE_REVIEWER = ROOT / ".claude" / "agents" / "code-reviewer.md"
HOOK_SOURCE = ROOT / "scripts" / "git-hooks"
TRACKED_HOOKS = ("post-checkout", "post-commit", "post-merge", "pre-push")
ZERO = "0" * 40
# git passes the remote name and its URL; the space is here so that a fake
# which loses argument boundaries (`$*` for `"$@"`) cannot pass.
PUSH_ARGUMENTS = ("origin", "file:///tmp/remote with space.git")

# Fakes: no network, no Claude, no real git-lfs. Each records its own argv and
# stdin as JSON onto one shared log, so a test can assert argument boundaries,
# byte-exact stdin, and the order the tools ran in.
RECORDER = """#!{python}
import json, os, sys

record = dict(
    tool="{tool}",
    argv=sys.argv[1:],
    stdin="" if sys.stdin.isatty() else sys.stdin.read(),
)
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
raise SystemExit(int(os.environ.get("{exit_variable}", "0")))
"""


def run(
    *command: str | Path,
    cwd: Path,
    env: dict[str, str] | None = None,
    input: str = "",
):
    return subprocess.run(
        [str(part) for part in command],
        cwd=cwd,
        env=env,
        input=input,
        text=True,
        capture_output=True,
        check=False,
    )


def git_repository(path: Path) -> None:
    assert run("git", "init", "-q", "-b", "main", cwd=path).returncode == 0
    assert run("git", "config", "user.name", "Hook Test", cwd=path).returncode == 0
    assert run("git", "config", "user.email", "hook@example.invalid", cwd=path).returncode == 0


def installer_env(root: Path) -> dict[str, str]:
    return {**os.environ, "JANKI_REPO_ROOT": str(root)}


def executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(0o755)
    return path


def stock_lfs_hook(name: str) -> str:
    """Exactly what `git lfs install` writes for `name` (git-lfs 3.7.1).

    Pinned here rather than derived from the installer so the two cannot drift
    together; `test_stock_hook_reference_matches_installed_git_lfs` checks this
    text against the real thing whenever git-lfs is available.
    """
    return (
        "#!/bin/sh\n"
        'command -v git-lfs >/dev/null 2>&1 || { printf >&2 "\\n%s\\n\\n" "This repository is '
        "configured for Git LFS but 'git-lfs' was not found on your path. If you no longer "
        f"wish to use Git LFS, remove this hook by deleting the '{name}' file in the hooks "
        "directory (set by 'core.hookspath'; usually '.git/hooks').\"; exit 2; }\n"
        f'git lfs {name} "$@"\n'
    )


def path_without_git_lfs(bin_dir: Path) -> str:
    """PATH that keeps git but cannot find git-lfs anywhere."""
    executable(bin_dir / "git", f'#!/bin/sh\nexec {shutil.which("git")} "$@"\n')
    entries = [
        entry
        for entry in os.environ["PATH"].split(os.pathsep)
        if entry and not (Path(entry) / "git-lfs").exists()
    ]
    return os.pathsep.join([str(bin_dir), *entries])


def recorder(path: Path, tool: str, exit_variable: str) -> Path:
    return executable(
        path,
        RECORDER.format(python=sys.executable, tool=tool, exit_variable=exit_variable),
    )


def fake_toolchain(tmp_path: Path, *, lfs_exit: int = 0) -> tuple[Path, dict[str, str]]:
    """A PATH whose git-lfs records instead of uploading, and no real Claude."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    recorder(bin_dir / "git-lfs", "git-lfs", "LFS_EXIT")
    recorder(bin_dir / "claude", "claude", "CLAUDE_EXIT")
    log = tmp_path / "fake-log.jsonl"
    env = {
        **os.environ,
        "FAKE_LOG": str(log),
        "LFS_EXIT": str(lfs_exit),
        # Nothing here may reach a paid model; a call is a test failure.
        "CLAUDE_EXIT": "1",
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    return log, env


def records(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def calls(log: Path, tool: str) -> list[dict]:
    return [record for record in records(log) if record["tool"] == tool]


def test_installer_puts_every_tracked_shim_in_a_fresh_clone(tmp_path: Path) -> None:
    git_repository(tmp_path)

    result = run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path))

    assert result.returncode == 0, result.stderr
    for name in TRACKED_HOOKS:
        installed = tmp_path / ".git" / "hooks" / name
        assert installed.read_bytes() == (HOOK_SOURCE / name).read_bytes()
        assert installed.stat().st_mode & stat.S_IXUSR
        assert f"git lfs {name}" in installed.read_text()
    for name in ("post-commit", "pre-push"):
        assert "scripts/janki-review.sh" in (tmp_path / ".git" / "hooks" / name).read_text()


def test_manual_code_reviewer_contract_excludes_japanese_content() -> None:
    contract = CODE_REVIEWER.read_text(encoding="utf-8")

    assert "Exclude\n`data/**` and `dist/**` from every diff" in contract
    assert "never judge whether authored Japanese" in contract
    assert "linguistically correct or natural" in contract
    assert "a later pass filters" not in contract


@pytest.mark.parametrize("conflicting", ["pre-push", "post-checkout"])
def test_installer_refuses_all_changes_when_one_hook_is_unmanaged(
    tmp_path: Path, conflicting: str
) -> None:
    git_repository(tmp_path)
    hooks = tmp_path / ".git" / "hooks"
    existing = executable(hooks / conflicting, "#!/bin/sh\necho mine\n")

    result = run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path))

    assert result.returncode == 1
    assert existing.read_text() == "#!/bin/sh\necho mine\n"
    untouched = [name for name in TRACKED_HOOKS if name != conflicting]
    assert not any((hooks / name).exists() for name in untouched), "installation is all-or-nothing"
    assert "unmanaged hook" in result.stderr


@pytest.mark.parametrize("stock", TRACKED_HOOKS)
def test_installer_replaces_a_stock_git_lfs_shim(tmp_path: Path, stock: str) -> None:
    """`git lfs install` wrote these; our shims call git-lfs themselves."""
    git_repository(tmp_path)
    hooks = tmp_path / ".git" / "hooks"
    executable(hooks / stock, stock_lfs_hook(stock))

    result = run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path))

    assert result.returncode == 0, result.stderr
    for name in TRACKED_HOOKS:
        assert (hooks / name).read_bytes() == (HOOK_SOURCE / name).read_bytes()


@pytest.mark.parametrize("edited", TRACKED_HOOKS)
def test_installer_refuses_a_git_lfs_shim_someone_appended_to(tmp_path: Path, edited: str) -> None:
    git_repository(tmp_path)
    hooks = tmp_path / ".git" / "hooks"
    body = stock_lfs_hook(edited) + 'echo "and my own thing" >&2\n'
    executable(hooks / edited, body)

    result = run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path))

    assert result.returncode == 1
    assert (hooks / edited).read_text() == body
    untouched = [name for name in TRACKED_HOOKS if name != edited]
    assert not any((hooks / name).exists() for name in untouched), "installation is all-or-nothing"
    assert "unmanaged hook" in result.stderr


@pytest.mark.skipif(shutil.which("git-lfs") is None, reason="git-lfs is not installed")
def test_stock_hook_reference_matches_installed_git_lfs(tmp_path: Path) -> None:
    git_repository(tmp_path)
    version = run("git", "lfs", "version", cwd=tmp_path).stdout
    if "git-lfs/3.7.1" not in version:
        pytest.skip(f"stock_lfs_hook is pinned to git-lfs 3.7.1; found {version.strip()!r}")

    assert run("git", "lfs", "install", "--local", cwd=tmp_path).returncode == 0

    for name in TRACKED_HOOKS:
        assert (tmp_path / ".git" / "hooks" / name).read_text() == stock_lfs_hook(name)


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


def content_review_repository(path: Path, *, mixed: bool = False) -> Path:
    git_repository(path)
    scripts = path / "scripts"
    scripts.mkdir()
    reviewer = scripts / "janki-review.sh"
    shutil.copy2(REVIEWER, reviewer)

    deck = path / "data" / "decks" / "lesson.yaml"
    deck.parent.mkdir(parents=True)
    deck.write_text("deck:\n  name: Lesson\n  description: first\n")
    if mixed:
        (path / "feature.py").write_text("VALUE = 1\n")
    assert run("git", "add", ".", cwd=path).returncode == 0
    assert run("git", "commit", "-qm", "first", cwd=path).returncode == 0

    deck.write_text("deck:\n  name: Lesson\n  description: second\n")
    if mixed:
        (path / "feature.py").write_text("VALUE = 2\n")
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

    log, env = fake_toolchain(tmp_path)
    post_commit = run(tmp_path / ".git" / "hooks" / "post-commit", cwd=tmp_path, env=env)
    assert post_commit.returncode == 0, post_commit.stderr
    assert not calls(log, "claude")
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


def test_content_only_range_never_starts_the_code_reviewer(tmp_path: Path) -> None:
    reviewer = content_review_repository(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    called = tmp_path / "claude-was-called"
    claude = fake_bin / "claude"
    claude.write_text('#!/bin/sh\ntouch "$CLAUDE_CALLED"\n')
    claude.chmod(0o755)
    env = {
        **os.environ,
        "CLAUDE_CALLED": str(called),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    result = run(
        reviewer,
        "gate",
        "HEAD~1..HEAD",
        "content-only",
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0
    assert "repository-content-only change" in result.stdout
    assert not called.exists()
    assert not (tmp_path / ".claude" / "reviews").exists()


def test_mixed_range_prompt_excludes_repository_content(tmp_path: Path) -> None:
    reviewer = content_review_repository(tmp_path, mixed=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    arguments = tmp_path / "claude-arguments"
    claude = fake_bin / "claude"
    claude.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$@" > "$CLAUDE_ARGUMENTS"\nprintf "VERDICT: CLEAN\\n"\n'
    )
    claude.chmod(0o755)
    env = {
        **os.environ,
        "CLAUDE_ARGUMENTS": str(arguments),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
    }

    result = run(
        reviewer,
        "gate",
        "HEAD~1..HEAD",
        "mixed",
        cwd=tmp_path,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    prompt = arguments.read_text()
    assert "git diff HEAD~1..HEAD -- . ':(exclude)data/**'" in prompt
    collapsed = " ".join(prompt.lower().split())
    assert "do not open or review any path under `data/`" in collapsed
    assert "linguistically correct or natural is explicitly outside" in collapsed


def push_repository(tmp_path: Path) -> SimpleNamespace:
    """Installed hooks, a recording reviewer, and the ref list git would send.

    The ref list carries an updated branch, a brand-new branch, and a branch
    deletion, so a hook that only handles the first line is visible.
    """
    reviewer = recorder(review_repository(tmp_path), "reviewer", "REVIEWER_EXIT")

    def sha(revision: str) -> str:
        return run("git", "rev-parse", revision, cwd=tmp_path).stdout.strip()

    head, parent = sha("HEAD"), sha("HEAD~1")
    assert run("git", "checkout", "-qb", "topic", cwd=tmp_path).returncode == 0
    (tmp_path / "topic.txt").write_text("topic\n")
    assert run("git", "add", "topic.txt", cwd=tmp_path).returncode == 0
    assert run("git", "commit", "-qm", "topic", cwd=tmp_path).returncode == 0
    topic = sha("HEAD")
    assert run("git", "checkout", "-q", "main", cwd=tmp_path).returncode == 0

    # Installed last: the setup commits and checkouts above must not run the
    # LFS hooks these install, which would demand a real git-lfs. The tests
    # invoke the hooks themselves, under fake_toolchain.
    assert run(INSTALLER, cwd=tmp_path, env=installer_env(tmp_path)).returncode == 0

    return SimpleNamespace(
        reviewer=reviewer,
        hooks=tmp_path / ".git" / "hooks",
        head=head,
        parent=parent,
        refs=(
            f"refs/heads/main {head} refs/heads/main {parent}\n"
            f"refs/heads/topic {topic} refs/heads/topic {ZERO}\n"
            f"refs/heads/gone {ZERO} refs/heads/gone {parent}\n"
        ),
        ranges=[f"{parent}..{head}", f"{head}..{topic}"],
    )


def test_pre_push_hands_git_lfs_the_push_verbatim_then_reviews_every_range(
    tmp_path: Path,
) -> None:
    repository = push_repository(tmp_path)
    log, env = fake_toolchain(tmp_path)

    result = run(
        repository.hooks / "pre-push",
        *PUSH_ARGUMENTS,
        cwd=tmp_path,
        env=env,
        input=repository.refs,
    )

    assert result.returncode == 0, result.stderr
    uploads = calls(log, "git-lfs")
    assert [upload["argv"] for upload in uploads] == [["pre-push", *PUSH_ARGUMENTS]]
    assert uploads[0]["stdin"] == repository.refs, "git-lfs decides what to upload from stdin"
    assert records(log)[0]["tool"] == "git-lfs", "objects must be uploaded before the slow review"
    reviews = calls(log, "reviewer")
    assert [review["argv"][1] for review in reviews] == repository.ranges
    assert [review["stdin"] for review in reviews] == ["", ""], "the reviewer must not eat refs"
    assert not calls(log, "claude")


@pytest.mark.parametrize("review", ["absent", "disabled"])
def test_pre_push_uploads_lfs_objects_even_when_nothing_is_reviewed(
    tmp_path: Path, review: str
) -> None:
    repository = push_repository(tmp_path)
    if review == "absent":
        repository.reviewer.unlink()
    else:
        shutil.copy2(REVIEWER, repository.reviewer)
        marker = tmp_path / ".claude" / "hooks" / "DISABLED"
        marker.parent.mkdir(parents=True)
        marker.write_text("budget pause\n")
    log, env = fake_toolchain(tmp_path)

    result = run(
        repository.hooks / "pre-push",
        *PUSH_ARGUMENTS,
        cwd=tmp_path,
        env=env,
        input=repository.refs,
    )

    assert result.returncode == 0, result.stderr
    uploads = calls(log, "git-lfs")
    assert [upload["argv"] for upload in uploads] == [["pre-push", *PUSH_ARGUMENTS]]
    assert uploads[0]["stdin"] == repository.refs
    assert not calls(log, "claude")


def test_pre_push_refuses_the_push_when_git_lfs_fails(tmp_path: Path) -> None:
    repository = push_repository(tmp_path)
    log, env = fake_toolchain(tmp_path, lfs_exit=1)

    result = run(
        repository.hooks / "pre-push",
        *PUSH_ARGUMENTS,
        cwd=tmp_path,
        env=env,
        input=repository.refs,
    )

    assert result.returncode != 0, "a push whose objects did not upload would be unusable"
    assert calls(log, "git-lfs")


def test_pre_push_refuses_the_push_when_git_lfs_is_missing(tmp_path: Path) -> None:
    repository = push_repository(tmp_path)
    bin_dir = tmp_path / "no-lfs-bin"
    bin_dir.mkdir()
    env = {
        **os.environ,
        "PATH": path_without_git_lfs(bin_dir),
        "FAKE_LOG": str(tmp_path / "no.log"),
    }

    result = run(
        repository.hooks / "pre-push",
        *PUSH_ARGUMENTS,
        cwd=tmp_path,
        env=env,
        input=repository.refs,
    )

    assert result.returncode != 0
    assert result.stderr.strip(), "the developer needs to be told why the push stopped"


def post_hook_arguments(repository: SimpleNamespace, name: str) -> tuple[str, ...]:
    """What git hands each hook: a checkout gets two shas and a branch flag."""
    return {
        "post-checkout": (repository.parent, repository.head, "1"),
        "post-commit": (),
        "post-merge": ("0",),
    }[name]


@pytest.mark.parametrize("name", ["post-checkout", "post-commit", "post-merge"])
def test_post_hook_forwards_its_arguments_to_git_lfs_without_a_reviewer(
    tmp_path: Path, name: str
) -> None:
    repository = push_repository(tmp_path)
    repository.reviewer.unlink()  # the review is optional; smudging files is not
    log, env = fake_toolchain(tmp_path)

    result = run(
        repository.hooks / name, *post_hook_arguments(repository, name), cwd=tmp_path, env=env
    )

    assert result.returncode == 0, result.stderr
    assert [call["argv"] for call in calls(log, "git-lfs")] == [
        [name, *post_hook_arguments(repository, name)]
    ]
    assert not calls(log, "claude")


@pytest.mark.parametrize("name", ["post-checkout", "post-commit", "post-merge"])
def test_post_hook_reports_a_git_lfs_failure_without_a_reviewer(tmp_path: Path, name: str) -> None:
    repository = push_repository(tmp_path)
    repository.reviewer.unlink()
    log, env = fake_toolchain(tmp_path, lfs_exit=2)

    result = run(
        repository.hooks / name, *post_hook_arguments(repository, name), cwd=tmp_path, env=env
    )

    assert result.returncode != 0, "files left as LFS pointers must not be silent"
    assert calls(log, "git-lfs")
