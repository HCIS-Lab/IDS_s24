#!/bin/bash

export MODEL=${1}
export ROOT_DIR=${2}
export FORECAST_TIME=${3}
export VALIDATE=${4}

export TORCH_DISTRIBUTED_DEBUG=INFO

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
python train.py --id ${MODEL}_${FORECAST_TIME} --batch_size 128 \
 --root_dir ${ROOT_DIR} \
 --logdir . --cpu_cores 20 --model $MODEL --validate $VALIDATE --forecast_time $FORECAST_TIME