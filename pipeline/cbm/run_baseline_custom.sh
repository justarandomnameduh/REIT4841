#!/bin/bash
# Train LaBo with custom baseline concepts

cd /home/nqmtien/REIT4841/pipeline/repos/LaBo

python main.py \
    --cfg /home/nqmtien/REIT4841/pipeline/cbm/configs/ham10000_baseline_custom.py \
    --work-dir /home/nqmtien/REIT4841/pipeline/cbm/results/ham10000_baseline_custom \
    --func asso_opt_main \
    2>&1 | tee /home/nqmtien/REIT4841/pipeline/cbm/logs/ham10000_baseline_custom.log
