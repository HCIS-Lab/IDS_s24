#!/bin/bash

export MODEL=${1:-constant}
export FORECAST_TIME=${2:-1}
export VALIDATE=${3:-0}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hcis-server/miniconda3/lib
export TORCH_DISTRIBUTED_DEBUG=INFO

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
python train.py --id ${MODEL}_${FORECAST_TIME} --batch_size 128 \
 --root_dir /media/hcis-s15/ssd2/SRL_dataset/data/HW1_dataset \
 --logdir . --cpu_cores 20 --model $MODEL --validate $VALIDATE --forecast_time $FORECAST_TIME