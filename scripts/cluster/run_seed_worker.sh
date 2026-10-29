#!/bin/bash
WID=$1 NW=3
cd "$(dirname "$0")/../.." || exit 1
source /opt/anaconda3/etc/profile.d/conda.sh
RUN="conda run --no-capture-output -n dv4 python -u"
export DUALV4_DEVICE=cuda:0 DUALV4_WINDOW=15 DUALV4_EPOCHS=100 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=xchk_logs/SEED_RUN_${WID}.log; echo "== seedworker$WID start $(date) ==" >> "$L"
n=0
while IFS="|" read -r DS M HEAD S; do
  n=$((n+1)); [ $(( (n-1) % NW )) -ne $WID ] && continue
  TAG="s${S}${HEAD}"
  if [ "$HEAD" = vae ]; then EX="DUALV4_VAE=1 DUALV4_NOKL=1"; else EX=""; fi
  if [ "$DS" = SMD ]; then OUT="output/SMD/$M/$TAG/epoch_0.1_lr_baseline/anomaly_scores.pkl"; EXP="experiments/SMD/$M/$TAG"; OUTD="output/SMD/$M/$TAG"; export DUALV4_SMD_MACHINE=$M
  else OUT="output/$DS/$TAG/epoch_0.1_lr_baseline/anomaly_scores.pkl"; EXP="experiments/$DS/$TAG"; OUTD="output/$DS/$TAG"; fi
  [ -f "$OUT" ] && { echo "[$(date +%H:%M:%S)] SKIP $DS/$M/$TAG" >> "$L"; continue; }
  export DUALV4_DS=$DS DUALV4_TAG=$TAG DUALV4_SEED=$S
  for kv in $EX; do export "$kv"; done
  echo "[$(date +%H:%M:%S)] START $DS/$M/$TAG" >> "$L"
  lf="xchk_logs/SEED_${DS}_${M}_${TAG}"
  if $RUN train_dualpath.py > "${lf}_train.log" 2>&1 && $RUN test_dualpath.py --grid 200 > "${lf}_test.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE $DS/$M/$TAG" >> "$L"; else echo "[$(date +%H:%M:%S)] FAIL $DS/$M/$TAG" >> "$L"; fi
  rm -rf "$EXP" 2>/dev/null
  find "$OUTD" -maxdepth 1 -type d -name "epoch_*" ! -name "epoch_0.1_lr_baseline" -exec rm -rf {} + 2>/dev/null
  for kv in $EX; do unset "${kv%%=*}"; done; unset DUALV4_SMD_MACHINE
done < jobs_seed.txt
echo "== seedworker$WID DONE $(date) ==" >> "$L"
