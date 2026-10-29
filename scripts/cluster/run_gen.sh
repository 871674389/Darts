#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1
source /opt/anaconda3/etc/profile.d/conda.sh
RUN="conda run --no-capture-output -n dv4 python -u"
export DUALV4_DEVICE=cuda:0 DUALV4_EPOCHS=100 DUALV4_WINDOW=15 DUALV4_SEED=42
L=xchk_logs/GEN_RUN.log
echo "===== GEN run start $(date) =====" >> "$L"
run_one(){
  local DS=$1 TAG=$2 EXTRA=$3
  export DUALV4_DS=$DS DUALV4_TAG=$TAG
  for kv in $EXTRA; do export "$kv"; done
  echo "[$(date +%H:%M:%S)] START $DS/$TAG [$EXTRA]" >> "$L"
  if $RUN train_dualpath.py > "xchk_logs/${DS}_${TAG}_$(echo $EXTRA|tr " /=" "___")_train.log" 2>&1 \
     && $RUN test_dualpath.py --grid 200 > "xchk_logs/${DS}_${TAG}_$(echo $EXTRA|tr " /=" "___")_test.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] DONE $DS/$TAG [$EXTRA]" >> "$L"
  else
    echo "[$(date +%H:%M:%S)] FAIL $DS/$TAG [$EXTRA]" >> "$L"
  fi
  for kv in $EXTRA; do unset "${kv%%=*}"; done
}
# Phase A: SMD 12机 + PSM, 预测头, seed42
for m in machine-1-1 machine-1-6 machine-1-7 machine-2-1 machine-2-2 machine-2-7 machine-2-8 machine-3-3 machine-3-4 machine-3-6 machine-3-8 machine-3-11; do
  run_one SMD s42pred "DUALV4_SMD_MACHINE=$m"
done
run_one PSM s42pred ""
echo "===== GEN PhaseA DONE $(date) =====" >> "$L"
