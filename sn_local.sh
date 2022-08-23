#!/bin/bash
#
run_id=dropout_40
model_id=local_a
data_source=model_prep
log_dir=log/${run_id}_${model_id}
mkdir -p ${log_dir}
snakemake --printshellcmds --keep-going --cores all --jobs 1 --rerun-incomplete 3_train/out/${data_source}/${run_id}/${model_id}_weights.pt 2>&1 | tee ${log_dir}/run.out
