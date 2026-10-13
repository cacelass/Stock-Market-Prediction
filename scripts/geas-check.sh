#!/usr/bin/env bash
set -euo pipefail
URL=${GEAS_URL:-http://192.168.1.145:8792}
TOKEN=${GEAS_TOKEN:-}
if [[ -z "$TOKEN" ]]; then
  echo "Falta GEAS_TOKEN"
  exit 1
fi
curl -s -H "Authorization: Bearer $TOKEN" "$URL/health"
echo
