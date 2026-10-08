#!/usr/bin/env bash
# Reads the BLOCKING line from this review's summary and asks Claude to fix blocking items.
# Capped at MAX_ROUNDS automated fix rounds per PR, then hands off via needs-human.
# If every blocking finding is in .github/workflows/, hands off right away: automated
# runs can't push workflow changes.
#
# Env: GH_TOKEN, REPO, PR_NUMBER, SINCE (review start, ISO 8601 UTC), FIX_MSG, MAX_ROUNDS.
set -euo pipefail

FIX_MSG=${FIX_MSG:-"@claude fix the blocking items in the latest review"}
MAX_ROUNDS=${MAX_ROUNDS:-3}

if [ -z "${GH_TOKEN:-}" ]; then
  echo "::error::PAT_TOKEN secret is not set"; exit 1
fi

comments=$(gh api --paginate --slurp "repos/$REPO/issues/$PR_NUMBER/comments")

# The latest bot comment from this review run that carries a BLOCKING line.
summary=$(printf '%s' "$comments" | jq -r --arg since "$SINCE" '
  [add[] | select(.user.type == "Bot" and .created_at >= $since) | .body | gsub("\r"; "")
   | select(test("(^|\n)BLOCKING: [0-9]+(\n|$)"))] | last // empty')

count=$(printf '%s\n' "$summary" | { grep -oE '^BLOCKING: [0-9]+$' || true; } | tail -n 1 | cut -d' ' -f2)
if [ -z "$count" ]; then
  echo "::warning::No 'BLOCKING: <count>' line in this review's summary; skipping auto-fix."; exit 0
fi
echo "Blocking findings: $count"
[ "$count" -gt 0 ] || exit 0

# One `BLOCKING_FILE: <path>` line per blocking finding (`-` if not tied to a file).
files=$(printf '%s\n' "$summary" | { grep -E '^BLOCKING_FILE: ' || true; } \
  | cut -d' ' -f2- | tr -d '`' | sed 's/[[:space:]]*$//')
file_count=$(printf '%s' "$files" | { grep -c '' || true; })
if [ "$file_count" -eq "$count" ] && ! printf '%s\n' "$files" | grep -qv '^\.github/workflows/'; then
  echo "All blocking findings are in .github/workflows/; handing off."
  gh pr comment "$PR_NUMBER" --repo "$REPO" --body "All $count blocking finding(s) are in \`.github/workflows/\`, which automated runs can't push. Handing off to a human."
  gh pr edit "$PR_NUMBER" --repo "$REPO" --add-label needs-human
  exit 0
fi

rounds=$(printf '%s' "$comments" | jq -r 'add[].body | gsub("\r"; "")' | { grep -cxF "$FIX_MSG" || true; })
echo "Previous automated fix rounds: $rounds"
if [ "$rounds" -ge "$MAX_ROUNDS" ]; then
  gh pr comment "$PR_NUMBER" --repo "$REPO" --body "Review still reports $count blocking finding(s) after $MAX_ROUNDS automated fix rounds. Handing off to a human."
  gh pr edit "$PR_NUMBER" --repo "$REPO" --add-label needs-human
  exit 0
fi
gh pr comment "$PR_NUMBER" --repo "$REPO" --body "$FIX_MSG"
