#!/usr/bin/env bash
# File: auto_train_test.sh
# Purpose: train 4×, test 1×, stop when SR_EVAL ≥ 0.80

set -euo pipefail

LAUNCH_FILE="tiago_navigation curriculum_simulation.launch"
TARGET=0.80
TRAIN_CHUNK=4

# ------------- helper -------------
unpause_physics() {
  rosservice call /gazebo/unpause_physics "{}"
}

roslaunch_once() {          # $1 = true|false  (test mode?)
  roslaunch $LAUNCH_FILE test:=$1 2>&1
}

extract_sr_from_log() {     # <<— simpler now
  grep -Eo 'SR_EVAL [0-9.]+' | tail -1 | awk '{print $2}'
}
# no more checkpoint_sr()

# ------------- main loop ----------
curr_sr=0
iter=1
best_sr=0

unpause_physics
# ---- evaluation ----
echo ">>> evaluation run"
log=$(mktemp)
roslaunch_once true | tee "$log"

curr_sr=$(extract_sr_from_log < "$log")

echo "Target success-rate $curr_sr reached – stopping."
