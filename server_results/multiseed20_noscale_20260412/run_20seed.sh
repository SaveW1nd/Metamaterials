#!/usr/bin/env bash
set -u
cd /root/shared-nvme/Metamaterials_multiseed20_noscale_20260412 || exit 1
SEEDS=$(seq 20260325 20260344)
MAX_PARALLEL=3
STATUS_FILE="$PWD/status.tsv"
: > "$STATUS_FILE"
ts() { date "+%F %T"; }
run_one() {
  local seed="$1"
  local cfg="configs/generated/train_seed_${seed}.yaml"
  local outdir="artifacts/checkpoints_seed_${seed}"
  local train_log="seed_${seed}.train.log"
  local eval_log="seed_${seed}.eval.log"
  mkdir -p "$outdir"
  echo -e "${seed}\tSTART\t$(ts)" >> "$STATUS_FILE"
  python3 run_train.py --config "$cfg" > "$train_log" 2>&1
  local train_code=$?
  echo -e "${seed}\tTRAIN_EXIT_${train_code}\t$(ts)" >> "$STATUS_FILE"
  if [[ $train_code -eq 0 ]]; then
    python3 run_eval.py --checkpoint "$outdir/best_model.pt" --data-dir artifacts/demo_dataset_no_input_scale > "$outdir/test_eval.json" 2> "$eval_log"
    local eval_code=$?
    echo -e "${seed}\tEVAL_EXIT_${eval_code}\t$(ts)" >> "$STATUS_FILE"
  fi
}
for seed in $SEEDS; do
  while [[ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]]; do
    sleep 10
  done
  run_one "$seed" &
  echo -e "${seed}\tLAUNCHED\t$(ts)" >> "$STATUS_FILE"
done
wait
printf "%b" "ALL_DONE\t$(ts)\n" >> "$STATUS_FILE"
