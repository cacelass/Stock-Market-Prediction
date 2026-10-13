#!/usr/bin/env bash
# geas-remote.sh — activa entorno remoto GEAS (instancia inversión) para portátil
# Uso: source scripts/geas-remote.sh  (o . scripts/geas-remote.sh)
set -euo pipefail

export GEAS_URL="http://192.168.1.145:8792"
export GEAS_TOKEN="taHBDs6M0MIrmqlOW4wVh6KPsLrM1M7x"
export GEAS_ACTOR="Portatil"

mkdir -p ~/.config/geas
cat > ~/.config/geas/client.env << 'ENV'
GEAS_URL=http://192.168.1.145:8792
GEAS_TOKEN=taHBDs6M0MIrmqlOW4wVh6KPsLrM1M7x
GEAS_ACTOR=Portatil
ENV
chmod 600 ~/.config/geas/client.env

echo "[GEAS] Remote activo: URL=$GEAS_URL ACTOR=$GEAS_ACTOR"
HEALTH=$(curl -s -H "Authorization: Bearer $GEAS_TOKEN" "$GEAS_URL/health" 2>/dev/null || echo '{"status":"err"}')
echo "[GEAS] Health: $HEALTH"
