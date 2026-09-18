#!/usr/bin/env bash
# Unverified R1Pro template only. The supplied DROID dataset uses run_lingbot_norm.py.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "$script_dir/run_lingbot_norm.py" \
  --robot-config "$script_dir/../configs/robot_configs/r1pro.yaml" \
  --train-config "$script_dir/../configs/vla/r1pro_load20000h.yaml" \
  "$@"
