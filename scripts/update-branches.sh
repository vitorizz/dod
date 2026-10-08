#!/usr/bin/env bash
# Merges BASE into open PRs that are behind it, so CI and review rerun on current code.
# A merge conflict gets a comment listing the files plus needs-human; that label is removed
# on a later run once the conflict is gone.
#
# Env: GH_TOKEN, REPO, BASE. Run from a clone whose `origin` is REPO.
set -euo pipefail

if [ -z "${GH_TOKEN:-}" ]; then
  echo "::error::PAT_TOKEN secret is not set"; exit 1
fi
git fetch --quiet origin "$BASE"
failed=0
prs=$(gh pr list --repo "$REPO" --base "$BASE" --state open --limit 100 \
  --json number,headRefName,headRepositoryOwner,labels \
  --jq '.[] | [.number, .headRepositoryOwner.login, .headRefName, ([.labels[].name] | index("needs-human") != null)] | @tsv')
[ -n "$prs" ] || { echo "No open PRs against $BASE."; exit 0; }

# Conflict comments carry this marker so the label can be traced back to them.
MARKER='<!-- update-branches:conflict -->'

# Sets $rc and $conflicts. merge-tree exits 0 = clean, 1 = conflicts (file names
# after the tree OID), >1 = error.
check_conflicts() {
  rc=0
  conflicts=""
  git fetch --quiet origin "pull/$1/head" || { rc=2; return; }
  conflicts=$(git merge-tree --write-tree --name-only --no-messages "origin/$BASE" FETCH_HEAD) || rc=$?
}

# True if the most recent needs-human label on the PR was added right after a conflict
# comment from this workflow, i.e. not by the review cap or a blocked Claude run.
labeled_for_conflict() {
  gh api --paginate --slurp "repos/$REPO/issues/$1/timeline?per_page=100" \
    | jq -e --arg m "$MARKER" '
        reduce (add[]) as $e ({prev: null, hit: false};
          if $e.event == "labeled" and $e.label.name == "needs-human"
          then .hit = (.prev.event == "commented" and ((.prev.body // "") | contains($m)))
          else . end
          | .prev = $e)
        | .hit' > /dev/null
}

# Merge BASE into the PR branch in a temporary worktree (so .gitattributes merge
# drivers apply) and push it. Same-repository branches only. Returns non-zero if
# the merge or the push fails; leaves no worktree behind.
merge_locally() {
  local pr=$1 owner=$2 ref=$3 dir ok=1
  [ "$owner" = "${REPO%%/*}" ] || { echo "#$pr is from a fork; not merging locally."; return 1; }
  git fetch --quiet origin "pull/$pr/head" || return 1
  dir=$(mktemp -d)
  git worktree add --quiet --detach "$dir" FETCH_HEAD || { rm -rf "$dir"; return 1; }
  if git -C "$dir" -c user.name="dod-update-branches" -c user.email="noreply@github.com" \
       merge --quiet --no-edit "origin/$BASE" \
     && git -C "$dir" push --quiet origin "HEAD:refs/heads/$ref"; then
    ok=0
  else
    git -C "$dir" merge --abort 2>/dev/null || true
  fi
  git worktree remove --force "$dir"
  return $ok
}

# Read PRs on fd 3 so gh calls in the loop can't consume the list from stdin.
while IFS=$'\t' read -r -u 3 pr owner ref blocked; do
  # A conflict this workflow flagged may have been resolved since; clear the label if so.
  if [ "$blocked" = "true" ] && labeled_for_conflict "$pr"; then
    check_conflicts "$pr"
    if [ "$rc" -eq 0 ]; then
      echo "#$pr no longer conflicts with $BASE; removing needs-human."
      gh pr edit "$pr" --repo "$REPO" --remove-label needs-human
      gh pr comment "$pr" --repo "$REPO" --body "Merge conflict with \`$BASE\` is resolved; removed \`needs-human\`."
      blocked=false
    elif [ "$rc" -eq 1 ]; then
      echo "#$pr still conflicts with $BASE."
    else
      echo "::error::Couldn't check #$pr for conflicts."; failed=1
    fi
  fi
  if ! behind=$(gh api "repos/$REPO/compare/$BASE...$owner:$ref" --jq '.behind_by'); then
    echo "::error::Couldn't compare #$pr with $BASE."; failed=1; continue
  fi
  if [ "$behind" = "0" ]; then
    echo "#$pr is up to date."; continue
  fi
  # Handed to a human (unresolved conflict, review cap, blocked run); don't re-comment on every push.
  if [ "$blocked" = "true" ]; then
    echo "#$pr is $behind behind but labeled needs-human; skipping."; continue
  fi
  echo "#$pr is $behind behind $BASE; updating."
  if update_out=$(gh pr update-branch "$pr" --repo "$REPO" 2>&1); then
    printf '%s\n' "$update_out"
    continue
  fi
  printf '%s\n' "$update_out"
  check_conflicts "$pr"
  if [ "$rc" -eq 0 ] && grep -qi 'conflict' <<< "$update_out"; then
    # GitHub's server-side merge ignores .gitattributes merge drivers (union on
    # docs/decisions.md), so it can report a conflict that a local merge resolves.
    if merge_locally "$pr" "$owner" "$ref"; then
      echo "#$pr merged with $BASE locally (GitHub's merge conflicted; .gitattributes resolved it)."
      continue
    fi
    echo "::error::Updating #$pr failed, and merging it locally failed too; check the log above."
    failed=1; continue
  fi
  if [ "$rc" -ne 1 ]; then
    echo "::error::Updating #$pr failed, and it isn't a merge conflict; check the log above."
    failed=1; continue
  fi
  files=$(printf '%s\n' "$conflicts" | tail -n +2 | sed '/^$/d' | sort -u | sed 's/^/- `/; s/$/`/')
  body="Couldn't update this branch with \`$BASE\`: merge conflict in:

$files

Resolve the conflicts (merge \`$BASE\` into this branch) and push; CI and review will rerun.
\`needs-human\` is removed automatically on the next push to \`$BASE\` once the conflict is gone.

$MARKER"
  gh pr comment "$pr" --repo "$REPO" --body "$body"
  gh pr edit "$pr" --repo "$REPO" --add-label needs-human
done 3<<< "$prs"
exit $failed
