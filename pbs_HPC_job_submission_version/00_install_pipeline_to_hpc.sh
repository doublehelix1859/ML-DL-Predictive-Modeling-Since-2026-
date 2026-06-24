#!/bin/bash
set -euo pipefail

# Run this from the repository root after editing config.env.
if [[ ! -f config.env ]]; then
  echo "ERROR: config.env not found. Copy config_template.env to config.env and edit it first."
  exit 1
fi
source config.env

mkdir -p "$ROOT" "$ROOT/logs" "$ROOT/input"
cp config.env "$ROOT/config.env"
cp scripts/*.py "$ROOT/"
cp pbs/step*.pbs "$ROOT/"
cp pbs/submit_all_steps.sh "$ROOT/submit_all_steps.sh"
chmod +x "$ROOT/submit_all_steps.sh"

echo "Installed pipeline files to: $ROOT"
echo "Next: cd $ROOT && bash submit_all_steps.sh"
