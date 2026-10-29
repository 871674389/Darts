#!/bin/bash
cd "$(dirname "$0")/../.." || exit 1; source /opt/anaconda3/etc/profile.d/conda.sh
export DUALV4_DEVICE=cuda:0 DUALV4_WINDOW=15 DUALV4_EPOCHS=100 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="conda run --no-capture-output -n dv4 python -u"
# SMD machine-1-1 两头
DUALV4_DS=SMD DUALV4_SMD_MACHINE=machine-1-1 DUALV4_TAG=MBpred $R train_dualpath.py > xchk_logs/MB_smd_pred.log 2>&1
DUALV4_DS=SMD DUALV4_SMD_MACHINE=machine-1-1 DUALV4_VAE=1 DUALV4_NOKL=1 DUALV4_TAG=MBvae $R train_dualpath.py > xchk_logs/MB_smd_vae.log 2>&1
# PSM 两头
DUALV4_DS=PSM DUALV4_TAG=MBpred $R train_dualpath.py > xchk_logs/MB_psm_pred.log 2>&1
DUALV4_DS=PSM DUALV4_VAE=1 DUALV4_NOKL=1 DUALV4_TAG=MBvae $R train_dualpath.py > xchk_logs/MB_psm_vae.log 2>&1
# 生成synth_val
DUALV4_DS=SMD DUALV4_SMD_MACHINE=machine-1-1 $R make_synth_val_gen.py > xchk_logs/MB_synth_smd.log 2>&1
DUALV4_DS=PSM $R make_synth_val_gen.py > xchk_logs/MB_synth_psm.log 2>&1
echo "MB_TRAIN_DONE $(date)" > xchk_logs/MB_STATUS.log
