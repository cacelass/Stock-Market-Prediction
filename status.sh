#!/usr/bin/env bash
# status.sh — ¿se puede trabajar en este repositorio? (spec §25)
#   ./status.sh          → READY TO WORK / NOT READY
#   ./status.sh --json   → salida estructurada para agentes
set -uo pipefail

export GEAS_DATA_DIR="${GEAS_DATA_DIR:-/home/cacelas/.local/share/geas-inversion}"
export GEAS_INSTANCE="${GEAS_INSTANCE:-inversion}"

LOCK="${GEAS_DATA_DIR}/geas.db.server"
if [[ -z "${GEAS_URL:-}" && -f "${LOCK}" ]]; then
  export GEAS_URL="${GEAS_URL:-http://127.0.0.1:8792}"
  export GEAS_TOKEN="${GEAS_TOKEN:-taHBDs6M0MIrmqlOW4wVh6KPsLrM1M7x}"
  export GEAS_ACTOR="${GEAS_ACTOR:-Raspy}"
fi

if [[ -n "${GEAS_URL:-}" && -n "${GEAS_TOKEN:-}" ]]; then
  if command -v geas >/dev/null 2>&1; then
    exec geas work status "${PWD}" "$@" --remote "${GEAS_URL}"
  else
    exec uv run --project "$(git rev-parse --show-toplevel)/.." python -m geas.work status "${PWD}" "$@" --remote "${GEAS_URL}"
  fi
fi

if command -v geas >/dev/null 2>&1; then
  exec geas work status "${PWD}" "$@"
else
  exec uv run --project "$(git rev-parse --show-toplevel)/.." python -m geas.work status "${PWD}" "$@"
fi
