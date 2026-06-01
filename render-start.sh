#!/bin/sh
set -e

if [ ! -f /etc/secrets/config.yaml ]; then
  echo "ERROR: Render secret file /etc/secrets/config.yaml was not found."
  echo "Create a Secret File named config.yaml in the Render service Environment tab."
  exit 1
fi

cp /etc/secrets/config.yaml /app/config.yaml

exec .venv/bin/python api/app.py --host 0.0.0.0 --port "${PORT:-10000}"
