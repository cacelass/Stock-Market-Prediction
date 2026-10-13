#!/usr/bin/env bash
# install-client-geas-inversion.sh — configura ESTE dispositivo para instancia GEAS inversión
# Uso: ./scripts/install-client-geas-inversion.sh [url] [token] [actor]
set -euo pipefail

URL="${1:-http://192.168.1.145:8792}"
TOKEN="${2:-taHBDs6M0MIrmqlOW4wVh6KPsLrM1M7x}"
ACTOR="${3:-Portatil}"

CONFIG_DIR="$HOME/.config/geas"
mkdir -p "$CONFIG_DIR"
umask 077
cat > "$CONFIG_DIR/client.env" <<EOF
# Cliente GEAS (instancia inversión) — este dispositivo
GEAS_URL=$URL
GEAS_TOKEN=$TOKEN
GEAS_ACTOR=$ACTOR
EOF
chmod 600 "$CONFIG_DIR/client.env"

echo "Configurado: $CONFIG_DIR/client.env"
echo "GEAS_URL=$URL"
echo "GEAS_ACTOR=$ACTOR"

if command -v curl >/dev/null 2>&1; then
  code="$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "$URL/health")"
  if [[ "$code" == "200" ]]; then
    echo "Health: OK"
  else
    echo "Health: HTTP $code" >&2
    exit 1
  fi
fi
