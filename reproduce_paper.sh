#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-outputs/paper}
DEVICE=${DEVICE:-auto}            # auto uses CUDA when available
SOLVER=${SOLVER:-gurobi}
EPISODES=${EPISODES:-300}         # MAPPO/IPPO episodes over L-1 to L-15 (round-robin)
TEST_SEEDS=${TEST_SEEDS:-"200042 200043 200044 200045 200046 200047 200048 200049 200050 200051"}
SYSTEMS=${SYSTEMS:-"multi_tote rack_climbing"}

for SYSTEM in $SYSTEMS; do
  python main_train.py pretrain --instances all-small --systems "$SYSTEM" --train-instances 1000 \
    --val-instances 300 --epochs 30 --solver "$SOLVER" --device "$DEVICE" --output "$OUT/pretrain_$SYSTEM"
  python main_train.py pretrain --algorithm IPPO --instances all-small --systems "$SYSTEM" --train-instances 1000 \
    --val-instances 300 --epochs 30 --solver "$SOLVER" --device "$DEVICE" --output "$OUT/pretrain_ippo_$SYSTEM"

  python main_train.py train --algorithm MAPPO --instances all-large --systems "$SYSTEM" --episodes "$EPISODES" \
    --initial-checkpoint "$OUT/pretrain_$SYSTEM/model.pt" --device "$DEVICE" --output "$OUT/mappo_$SYSTEM"
  python main_train.py train --algorithm IPPO --instances all-large --systems "$SYSTEM" --episodes "$EPISODES" \
    --initial-checkpoint "$OUT/pretrain_ippo_$SYSTEM/model.pt" --device "$DEVICE" --output "$OUT/ippo_$SYSTEM"

  python main_train.py gap --instances all-small --systems "$SYSTEM" --seeds $TEST_SEEDS \
    --checkpoint "$OUT/pretrain_$SYSTEM/model.pt" --solver "$SOLVER" --device "$DEVICE" --output "$OUT/gap_$SYSTEM"

  python main_train.py evaluate --instances all-large --systems "$SYSTEM" --seeds $TEST_SEEDS \
    --methods MAPPO IPPO C-SGH R3 G3 --checkpoint "$OUT/mappo_$SYSTEM/model.pt" \
    --ippo-checkpoint "$OUT/ippo_$SYSTEM/model.pt" --device "$DEVICE" --output "$OUT/eval_$SYSTEM"

done
