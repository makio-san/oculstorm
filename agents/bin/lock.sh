#!/usr/bin/env bash
# One heavy job (training, tuning, holdout scoring) at a time on this machine.
#
# Usage:
#   bash agents/bin/lock.sh take <agent>      # exit 0 when taken, 1 when another agent holds it
#   bash agents/bin/lock.sh release <agent>
#
# A lock older than 6 hours is treated as left behind by a crashed run and is taken over.
set -euo pipefail

LOCK="${OCULSTORM_AGENTS_ROOT:-/Users/esma/Projects/oculstorm-agents}/heavy.lock"
ACTION="${1:?usage: lock.sh take|release <agent>}"
AGENT="${2:?usage: lock.sh take|release <agent>}"

case "$ACTION" in
  take)
    if mkdir "$LOCK" 2>/dev/null; then
      echo "$AGENT $(date -u +%Y-%m-%dT%H:%MZ)" > "$LOCK/owner"
      echo "lock taken by $AGENT"
      exit 0
    fi
    if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +360 2>/dev/null)" ]; then
      echo "$AGENT $(date -u +%Y-%m-%dT%H:%MZ) (took over a stale lock: $(cat "$LOCK/owner" 2>/dev/null))" > "$LOCK/owner"
      touch "$LOCK"
      echo "stale lock taken over by $AGENT"
      exit 0
    fi
    echo "lock held by $(cat "$LOCK/owner" 2>/dev/null || echo unknown)"
    exit 1
    ;;
  release)
    if [ -d "$LOCK" ] && grep -q "^$AGENT " "$LOCK/owner" 2>/dev/null; then
      rm -rf "$LOCK"
      echo "lock released"
    else
      echo "not holding the lock"
    fi
    ;;
  *)
    echo "usage: lock.sh take|release <agent>" >&2
    exit 2
    ;;
esac
