#!/usr/bin/env bash
# Give an agent a clean worktree of origin/main, outside the owner's own checkout.
#
# Usage: bash agents/bin/workspace.sh <agent>
# Prints the worktree path: $OCULSTORM_AGENTS_ROOT/work/<agent>
# (OCULSTORM_AGENTS_ROOT defaults to /Users/esma/Projects/oculstorm-agents; the orchestra
# sets it to /Users/esma/Projects/oculstorm_orchestra)
#
# Every run starts from the newest origin/main with no local changes. Ignored files
# (a downloaded parquet, data/raw) survive so a run that crashed can still clean them up.
set -euo pipefail

AGENT="${1:?usage: workspace.sh <agent>}"
ROOT="${OCULSTORM_AGENTS_ROOT:-/Users/esma/Projects/oculstorm-agents}"
BASE="$ROOT/base"
WORK="$ROOT/work/$AGENT"

mkdir -p "$ROOT/work" "$ROOT/logs"
if [ ! -d "$BASE/.git" ]; then
  git clone -q https://github.com/makio-san/oculstorm.git "$BASE"
fi
git -C "$BASE" fetch -q --prune origin
git -C "$BASE" worktree prune

if [ ! -e "$WORK/.git" ]; then
  git -C "$BASE" worktree add -q --detach "$WORK" origin/main
fi

cd "$WORK"
git checkout -q -f --detach origin/main
git reset -q --hard origin/main
git clean -fdq
echo "$WORK"
