#!/usr/bin/env bash
# File: auto_train_test.sh
# Purpose: train 4×, test 1×, stop when SR_EVAL ≥ 0.80
# Modified: added update_better_model flag (only on first iteration of 4-chunk loop after new best SR)

set -euo pipefail

LAUNCH_FILE="tiago_navigation curriculum_simulation.launch"
TARGET=0.7
TRAIN_CHUNK=4
THRESHOLD_LR_UPDATE=0.6  # if SR_EVAL > threshold, then update learning rate

# ------------- helper -------------
unpause_physics() {
  rosservice call /gazebo/unpause_physics "{}"
}

roslaunch_once() {          # $1 = true|false  (test mode?), optionally more args
  roslaunch $LAUNCH_FILE test:=$1 "${@:2}" 2>&1
}

extract_sr_from_log() {     # <<— simpler now
  grep -Eo 'SR_EVAL [0-9.]+' | tail -1 | awk '{print $2}'
}
# no more checkpoint_sr()

# ------------- main loop ----------
curr_sr=0.0
iter=1
best_sr=0.0
PENDING_UPDATE_BETTER_MODEL=0
UPDATE_LR=0

while (( $(echo "$curr_sr < $TARGET" | bc -l) )); do
  unpause_physics
  # ---- training ----
  for i in $(seq 1 $TRAIN_CHUNK); do

    echo ">>> training run $i/$TRAIN_CHUNK (best_sr=$best_sr)"
    roslaunch_once false 
    unpause_physics
  done

  # ---- evaluation ----
  echo ">>> evaluation run"
  log=$(mktemp)
  roslaunch_once true curr_sr:=$best_sr learning_rate_update:=$THRESHOLD_LR_UPDATE | tee "$log"

  curr_sr=$(extract_sr_from_log < "$log")
  if (( $(echo "$curr_sr > $best_sr" | bc -l) )); then
    best_sr=$curr_sr
    echo ">>> new best SR_EVAL: $best_sr"
    PENDING_UPDATE_BETTER_MODEL=1   # segnala che nella prossima iterazione il flag verrà attivato
  fi

  # Controlla se superare la soglia per l'update del learning rate
  if (( $(echo "$curr_sr >= $THRESHOLD_LR_UPDATE" | bc -l) )); then
    echo ">>> SR_EVAL ($curr_sr) >= threshold ($THRESHOLD_LR_UPDATE) - scheduling LR update"
    UPDATE_LR=1
    THRESHOLD_LR_UPDATE=$(echo "$THRESHOLD_LR_UPDATE + 0.1" | bc) # aumenta la soglia per il prossimo update
  fi
  echo ">>> current success-rate: $curr_sr at iteration $iter "
  echo ">>> Best success-rate so far: $best_sr"
  iter=$((iter + 1))
  rm "$log"
done

echo "Target success-rate $curr_sr reached – stopping."