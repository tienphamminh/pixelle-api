#!/bin/sh
set -e

if [ -n "${CONFIG_YAML:-}" ]; then
  printf "%s" "$CONFIG_YAML" > /app/config.yaml
elif [ ! -f /app/config.yaml ]; then
  echo "ERROR: config.yaml was not found."
  echo "Create a Hugging Face Space secret named CONFIG_YAML containing your config.yaml content."
  exit 1
fi

mkdir -p /app/output /app/data /app/temp

exec .venv/bin/python api/app.py --host 0.0.0.0 --port "${PORT:-7860}"
