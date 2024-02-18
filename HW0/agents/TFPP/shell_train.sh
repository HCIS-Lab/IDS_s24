#!/bin/bash

export CARLA_ROOT=/home/hcis-s15/Documents/projects/SRL/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPIz
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":${PYTHONPATH}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hcis-s15/miniconda3/lib

export OMP_NUM_THREADS=20  # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
torchrun --nnodes=1 --nproc_per_node=1 --max_restarts=0 --rdzv_id=42353467 --rdzv_backend=c10d train.py --id validate_garage --batch_size 4 --setting all --root_dir /media/hcis-s15/ssd2/SRL_dataset/data/training_data_final --logdir /media/hcis-s15/ssd2/tfuse_pp_ckpt --backbone bev_encoder --use_controller_input_prediction 1 --use_wp_gru 1 --use_discrete_command 1 --use_tp 1 --continue_epoch 0 --cpu_cores 20 --num_repetitions 1
