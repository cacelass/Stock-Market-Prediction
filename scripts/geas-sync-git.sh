#!/usr/bin/env bash
set -euo pipefail

BRANCH=$(git branch --show-current)
FEATURE=${1:-"SYNC"}
DATE=$(date +%Y-%m-%dT%H:%M:%S)

# Stash any uncommitted changes to avoid rebase blocking
if ! git diff --quiet || ! git diff --cached --quiet; then
  git stash push -m "${FEATURE} pre-sync ${DATE}" >/dev/null 2>&1 || true
  STASHED=1
else
  STASHED=0
fi

git pull --rebase origin "${BRANCH}"
git push origin "${BRANCH}"

if [ "${STASHED}" -eq 1 ]; then
  if git stash pop >/dev/null 2>&1; then
    :
  else
    echo "WARNING: stash pop tuvo conflictos, resuélvelos manualmente" >&2
    exit 1
  fi
fi
echo "SYNC OK"
