#!/bin/bash

export CARLA_ROOT=$1
export HOMEWORK_DIR=$2
export MODEL_NAME=$3

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${HOMEWORK_DIR}/DOS/leaderboard
export PYTHONPATH=$PYTHONPATH:${HOMEWORK_DIR}/DOS/scenario_runner

export LEADERBOARD_ROOT=${HOMEWORK_DIR}/DOS/leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000 # same as the carla server port
export TM_PORT=2500 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
export ROUTES=${HOMEWORK_DIR}/DOS/DOS_benchmark/DOS_01_town05.xml
export TEAM_AGENT=${HOMEWORK_DIR}/agents/${MODEL_NAME}/longest6_agent.py # agent
export TEAM_CONFIG=${HOMEWORK_DIR}/agents/checkpoints/${MODEL_NAME} # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=${HOMEWORK_DIR}/results/DOS/${MODEL_NAME}/DOS_01_result.json # results file
export SCENARIOS=${HOMEWORK_DIR}/DOS/DOS_benchmark/DOS_01_town05.json
export SAVE_PATH=${HOMEWORK_DIR}/results/DOS/${MODEL_NAME} # path for saving episodes while evaluating
export RESUME=True
export DIRECT=1

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT} \
--record=reocrd

