"""CI automation scripts in scripts/, run against a fake `gh` on PATH (real git and jq)."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
REPO = "o/r"
SINCE = "2026-10-07T12:00:00Z"
FIX_MSG = "@claude fix the blocking items in the latest review"

pytestmark = pytest.mark.skipif(
    not (shutil.which("bash") and shutil.which("jq") and shutil.which("git")),
    reason="needs bash, jq and git",
)

# Logs each call as a JSON argv line, then answers from the first rule whose `match`
# tokens all appear in argv. `--jq` is applied to the canned output with real jq.
FAKE_GH = """#!{python}
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as f:
    f.write(json.dumps(args) + "\\n")
with open(os.environ["FAKE_GH_RULES"]) as f:
    rules = json.load(f)
for rule in rules:
    if all(tok in args for tok in rule["match"]):
        out = rule.get("stdout", "")
        if out and "--jq" in args:
            out = subprocess.run(["jq", "-r", args[args.index("--jq") + 1]], input=out,
                                 capture_output=True, text=True, check=True).stdout
        sys.stdout.write(out)
        sys.stderr.write(rule.get("stderr", ""))
        sys.exit(rule.get("rc", 0))
"""


class FakeGh:
    def __init__(self, tmp_path):
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        gh = self.bin / "gh"
        gh.write_text(FAKE_GH.format(python=sys.executable))
        gh.chmod(0o755)
        self.log = tmp_path / "gh.log"
        self.log.touch()
        self.rules_path = tmp_path / "gh-rules.json"
        self.rules = []

    def on(self, *match, json_out=None, rc=0, stderr=""):
        stdout = "" if json_out is None else json.dumps(json_out)
        self.rules.append({"match": list(match), "stdout": stdout, "rc": rc, "stderr": stderr})

    def run(self, script, cwd, **env):
        self.rules_path.write_text(json.dumps(self.rules))
        full_env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "FAKE_GH_LOG": str(self.log),
            "FAKE_GH_RULES": str(self.rules_path),
            "GH_TOKEN": "t",
            "REPO": REPO,
            **env,
        }
        return subprocess.run(["bash", str(SCRIPTS / script)], cwd=cwd, env=full_env,
                              capture_output=True, text=True)

    @property
    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def comments(self):
        return [c[c.index("--body") + 1] for c in self.calls if c[:2] == ["pr", "comment"]]

    def labels_added(self):
        return [c[c.index("--add-label") + 1] for c in self.calls if "--add-label" in c]


@pytest.fixture
def gh(tmp_path):
    return FakeGh(tmp_path)


# --- review-auto-fix.sh -------------------------------------------------------

def comment(body, bot=True, at="2026-10-07T12:05:00Z"):
    return {"body": body, "user": {"type": "Bot" if bot else "User"}, "created_at": at}


def auto_fix(gh, tmp_path, comments):
    gh.on("api", f"repos/{REPO}/issues/5/comments", json_out=[comments])
    return gh.run("review-auto-fix.sh", tmp_path, PR_NUMBER="5", SINCE=SINCE,
                  FIX_MSG=FIX_MSG, MAX_ROUNDS="3")


def test_blocking_count_above_zero_requests_fix(gh, tmp_path):
    r = auto_fix(gh, tmp_path, [comment("## Review\n- bug\r\nBLOCKING_FILE: src/dod/x.py\r\nBLOCKING: 1\r\n")])
    assert r.returncode == 0, r.stderr
    assert gh.comments() == [FIX_MSG]
    assert gh.labels_added() == []


def test_blocking_zero_does_nothing(gh, tmp_path):
    r = auto_fix(gh, tmp_path, [comment("No issues found.\nBLOCKING: 0")])
    assert r.returncode == 0, r.stderr
    assert "Blocking findings: 0" in r.stdout
    assert gh.comments() == [] and gh.labels_added() == []


def test_missing_blocking_line_warns_and_skips(gh, tmp_path):
    r = auto_fix(gh, tmp_path, [
        comment("Old review\nBLOCKING: 2", at="2026-10-07T11:00:00Z"),  # before this run
        comment("Summary without the line", bot=True),
        comment("BLOCKING: 4", bot=False),  # not from the reviewer
    ])
    assert r.returncode == 0, r.stderr
    assert "No 'BLOCKING: <count>' line" in r.stdout
    assert gh.comments() == [] and gh.labels_added() == []


def test_no_comments_at_all_warns_and_skips(gh, tmp_path):
    r = auto_fix(gh, tmp_path, [])
    assert r.returncode == 0, r.stderr
    assert "No 'BLOCKING: <count>' line" in r.stdout
    assert gh.calls == [["api", "--paginate", "--slurp", f"repos/{REPO}/issues/5/comments"]]


def test_three_round_cap_hands_off(gh, tmp_path):
    prior = [comment(FIX_MSG + "\r\n", bot=False, at=f"2026-10-0{d}T00:00:00Z") for d in (4, 5, 6)]
    r = auto_fix(gh, tmp_path, prior + [comment("BLOCKING_FILE: src/a.py\nBLOCKING: 1")])
    assert r.returncode == 0, r.stderr
    assert "Previous automated fix rounds: 3" in r.stdout
    assert gh.comments() == ["Review still reports 1 blocking finding(s) after 3 automated fix rounds. "
                             "Handing off to a human."]
    assert gh.labels_added() == ["needs-human"]


def test_below_cap_still_requests_fix(gh, tmp_path):
    prior = [comment(FIX_MSG, bot=False, at=f"2026-10-0{d}T00:00:00Z") for d in (5, 6)]
    r = auto_fix(gh, tmp_path, prior + [comment("BLOCKING: 1")])
    assert r.returncode == 0, r.stderr
    assert gh.comments() == [FIX_MSG]


def test_all_findings_in_workflows_hands_off(gh, tmp_path):
    body = ("- a\n- b\nBLOCKING_FILE: `.github/workflows/ci.yml`\n"
            "BLOCKING_FILE: .github/workflows/update-branches.yml \nBLOCKING: 2")
    r = auto_fix(gh, tmp_path, [comment(body)])
    assert r.returncode == 0, r.stderr
    assert gh.comments() == ["All 2 blocking finding(s) are in `.github/workflows/`, which automated "
                             "runs can't push. Handing off to a human."]
    assert gh.labels_added() == ["needs-human"]


@pytest.mark.parametrize("files", [
    [".github/workflows/ci.yml", "src/dod/ledger.py"],  # one finding outside workflows
    [".github/workflows/ci.yml", "-"],                  # one finding not tied to a file
    [".github/workflows/ci.yml"],                       # fewer file lines than findings
])
def test_findings_outside_workflows_request_fix(gh, tmp_path, files):
    lines = "".join(f"BLOCKING_FILE: {f}\n" for f in files)
    r = auto_fix(gh, tmp_path, [comment(f"{lines}BLOCKING: 2")])
    assert r.returncode == 0, r.stderr
    assert gh.comments() == [FIX_MSG]
    assert gh.labels_added() == []


def test_missing_token_fails(gh, tmp_path):
    r = gh.run("review-auto-fix.sh", tmp_path, GH_TOKEN="", PR_NUMBER="5", SINCE=SINCE)
    assert r.returncode == 1
    assert "PAT_TOKEN secret is not set" in r.stdout
    assert gh.calls == []


# --- update-branches.sh -------------------------------------------------------

def git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(cwd), *args],
                          check=True, capture_output=True, text=True).stdout


def commit(work, files, msg):
    for name, text in files.items():
        (work / name).write_text(text)
    git(work, "add", "-A")
    git(work, "commit", "-q", "-m", msg)


@pytest.fixture
def repo(tmp_path):
    """Bare origin with `main` one commit ahead of PR #7 (refs/pull/7/head); returns a clone."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(tmp_path, "init", "-q", "-b", "main", str(work))
    git(work, "remote", "add", "origin", str(origin))
    commit(work, {"a.txt": "base\n", "b.txt": "base\n"}, "base")
    git(work, "push", "-q", "origin", "main")

    def make_pr(pr_files):
        git(work, "checkout", "-q", "-b", "feature")
        commit(work, pr_files, "pr")
        git(work, "push", "-q", "origin", "HEAD:refs/pull/7/head")
        git(work, "checkout", "-q", "main")
        commit(work, {"a.txt": "main\n"}, "main moves on")
        git(work, "push", "-q", "origin", "main")
        clone = tmp_path / "clone"
        git(tmp_path, "clone", "-q", str(origin), str(clone))
        return clone

    return make_pr


def pr_list(gh, labels=()):
    gh.on("pr", "list", json_out=[{
        "number": 7, "headRefName": "feature", "headRepositoryOwner": {"login": "o"},
        "labels": [{"name": n} for n in labels],
    }])
    gh.on("api", f"repos/{REPO}/compare/main...o:feature", json_out={"behind_by": 1})


def test_clean_branch_update(gh, repo):
    clone = repo({"b.txt": "feature\n"})
    pr_list(gh)
    gh.on("pr", "update-branch", "7")
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "#7 is 1 behind main; updating." in r.stdout
    assert ["pr", "update-branch", "7", "--repo", REPO] in gh.calls
    assert gh.comments() == [] and gh.labels_added() == []


def test_merge_conflict_comments_and_labels(gh, repo):
    clone = repo({"a.txt": "feature\n", "b.txt": "feature\n"})
    pr_list(gh)
    gh.on("pr", "update-branch", "7", rc=1, stderr="merge conflict\n")
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 0, r.stdout + r.stderr
    [body] = gh.comments()
    assert "merge conflict in:\n\n- `a.txt`\n\nResolve" in body
    assert "b.txt" not in body
    assert body.endswith("<!-- update-branches:conflict -->")
    assert gh.labels_added() == ["needs-human"]


def test_update_failure_without_conflict_errors(gh, repo):
    clone = repo({"b.txt": "feature\n"})
    pr_list(gh)
    gh.on("pr", "update-branch", "7", rc=1, stderr="permission denied\n")
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 1
    assert "isn't a merge conflict" in r.stdout
    assert gh.comments() == [] and gh.labels_added() == []


def test_up_to_date_pr_is_left_alone(gh, repo):
    clone = repo({"b.txt": "feature\n"})
    gh.on("api", f"repos/{REPO}/compare/main...o:feature", json_out={"behind_by": 0})
    pr_list(gh)
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "#7 is up to date." in r.stdout
    assert not any(c[:2] == ["pr", "update-branch"] for c in gh.calls)


# --- update-branches.sh: needs-human handling ------------------------------------

MARKER = "<!-- update-branches:conflict -->"


def timeline(gh, pr, events):
    # `gh api --paginate --slurp` returns a list of pages.
    gh.on("api", "--slurp", f"repos/{REPO}/issues/{pr}/timeline?per_page=100", json_out=[events])


def conflict_comment():
    return {"event": "commented", "body": f"Couldn't update this branch...\n\n{MARKER}"}


def labeled(name="needs-human"):
    return {"event": "labeled", "label": {"name": name}}


def labels_removed(gh):
    return [c[c.index("--remove-label") + 1] for c in gh.calls if "--remove-label" in c]


def test_resolved_conflict_removes_label_then_updates(gh, repo):
    clone = repo({"b.txt": "feature\n"})  # no longer conflicts with main
    pr_list(gh, labels=["needs-human"])
    timeline(gh, 7, [conflict_comment(), labeled()])
    gh.on("pr", "update-branch", "7")
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "#7 no longer conflicts with main; removing needs-human." in r.stdout
    assert labels_removed(gh) == ["needs-human"]
    assert gh.comments() == ["Merge conflict with `main` is resolved; removed `needs-human`."]
    # Unblocked in the same run, so the behind branch is updated right away.
    assert ["pr", "update-branch", "7", "--repo", REPO] in gh.calls


def test_still_conflicting_keeps_label_and_skips(gh, repo):
    clone = repo({"a.txt": "feature\n"})  # still conflicts on a.txt
    pr_list(gh, labels=["needs-human"])
    timeline(gh, 7, [conflict_comment(), labeled()])
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "#7 still conflicts with main." in r.stdout
    assert "#7 is 1 behind but labeled needs-human; skipping." in r.stdout
    assert labels_removed(gh) == [] and gh.comments() == []
    assert not any(c[:2] == ["pr", "update-branch"] for c in gh.calls)


@pytest.mark.parametrize("events", [
    # Labeled by the review cap: preceded by a comment without the marker.
    [{"event": "commented", "body": "Handing off to a human."}, labeled()],
    # Labeled for a conflict once, then removed and relabeled for another reason.
    [conflict_comment(), labeled(), {"event": "unlabeled", "label": {"name": "needs-human"}},
     {"event": "commented", "body": "Review still reports 1 blocking finding(s)"}, labeled()],
    # The marker comment is not immediately before the label.
    [conflict_comment(), {"event": "committed"}, labeled()],
    # A conflict comment followed by a different label.
    [conflict_comment(), labeled("bug")],
    # No events at all.
    [],
])
def test_needs_human_for_other_reasons_is_never_removed(gh, repo, events):
    clone = repo({"b.txt": "feature\n"})  # mergeable, so only the timeline check protects the label
    pr_list(gh, labels=["needs-human"])
    timeline(gh, 7, events)
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 0, r.stdout + r.stderr
    assert labels_removed(gh) == [] and gh.comments() == []
    assert "#7 is 1 behind but labeled needs-human; skipping." in r.stdout
    assert not any(c[:2] == ["pr", "update-branch"] for c in gh.calls)


def test_conflict_check_error_fails_without_touching_label(gh, repo):
    clone = repo({"b.txt": "feature\n"})
    gh.on("pr", "list", json_out=[{  # PR #8 has no refs/pull/8/head, so the fetch fails
        "number": 8, "headRefName": "feature", "headRepositoryOwner": {"login": "o"},
        "labels": [{"name": "needs-human"}],
    }])
    gh.on("api", f"repos/{REPO}/compare/main...o:feature", json_out={"behind_by": 1})
    timeline(gh, 8, [conflict_comment(), labeled()])
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 1
    assert "Couldn't check #8 for conflicts." in r.stdout
    assert labels_removed(gh) == [] and gh.comments() == []
    assert not any(c[:2] == ["pr", "update-branch"] for c in gh.calls)


# --- update-branches.sh: local merge when GitHub's merge ignores .gitattributes ----

@pytest.fixture
def union_repo(tmp_path):
    """Like `repo`, but both sides append to a file merged with git's union driver."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    git(tmp_path, "init", "-q", "-b", "main", str(work))
    git(work, "remote", "add", "origin", str(origin))
    commit(work, {".gitattributes": "log.md merge=union\n", "log.md": "- base\n"}, "base")
    git(work, "push", "-q", "origin", "main")
    git(work, "checkout", "-q", "-b", "feature")
    commit(work, {"log.md": "- base\n- from pr\n"}, "pr")
    git(work, "push", "-q", "origin", "HEAD:refs/pull/7/head", "HEAD:refs/heads/feature")
    git(work, "checkout", "-q", "main")
    commit(work, {"log.md": "- base\n- from main\n"}, "main moves on")
    git(work, "push", "-q", "origin", "main")
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(origin), str(clone))
    return origin, clone


def test_union_only_conflict_is_merged_locally_and_pushed(gh, union_repo):
    origin, clone = union_repo
    pr_list(gh)
    gh.on("pr", "update-branch", "7", rc=1, stderr="merge conflict\n")
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "#7 merged with main locally" in r.stdout
    assert gh.comments() == [] and gh.labels_added() == []
    log = git(origin, "show", "feature:log.md")
    assert "- from pr" in log and "- from main" in log
    # The pushed branch contains main, so it's no longer behind.
    assert git(origin, "merge-base", "--is-ancestor", "main", "feature") == ""
    # No temporary worktree is left registered.
    assert git(clone, "worktree", "list").count("\n") == 1


def test_local_merge_push_failure_errors_without_labeling(gh, union_repo):
    origin, clone = union_repo
    pr_list(gh)
    gh.on("pr", "update-branch", "7", rc=1, stderr="merge conflict\n")
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\necho rejected >&2\nexit 1\n")
    hook.chmod(0o755)
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 1
    assert "merging it locally failed too" in r.stdout
    assert gh.comments() == [] and gh.labels_added() == []
    assert git(clone, "worktree", "list").count("\n") == 1


def test_fork_prs_are_never_pushed_to(gh, union_repo):
    origin, clone = union_repo
    gh.on("pr", "list", json_out=[{
        "number": 7, "headRefName": "feature", "headRepositoryOwner": {"login": "someone-else"},
        "labels": [],
    }])
    gh.on("api", f"repos/{REPO}/compare/main...someone-else:feature", json_out={"behind_by": 1})
    gh.on("pr", "update-branch", "7", rc=1, stderr="merge conflict\n")
    before = git(origin, "rev-parse", "feature")
    r = gh.run("update-branches.sh", clone, BASE="main")
    assert r.returncode == 1
    assert "from a fork" in r.stdout
    assert git(origin, "rev-parse", "feature") == before
