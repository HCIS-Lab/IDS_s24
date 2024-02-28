export CARLA_ROOT=$1
export HOMEWORK_DIR=$2
export MODEL_NAME=$3

export LEADERBOARD_ROOT=${HOMEWORK_DIR}/leaderboard_2.0
export SCENARIO_RUNNER_ROOT=${LEADERBOARD_ROOT}/scenario_runner
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg":${PYTHONPATH}

export ROUTES=${LEADERBOARD_ROOT}/Data/ConstructionObstacleTwoWays.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CHECKPOINT_ENDPOINT=${HOMEWORK_DIR}/results/leaderboard_2.0/${MODEL_NAME}/leaderboard_2.0.json
export TEAM_AGENT=${HOMEWORK_DIR}/agents/${MODEL_NAME}/leaderboard_agent.py
export TEAM_CONFIG=${HOMEWORK_DIR}/agents/checkpoints/${MODEL_NAME}
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export SAVE_PATH=${HOMEWORK_DIR}/results/leaderboard_2.0/${MODEL_NAME}
export UNCERTAINTY_THRESHOLD=0.33
export DIRECT=1



#!/bin/sh


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--routes=${ROUTES} \
--routes-subset=${ROUTES_SUBSET} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--debug-checkpoint=${DEBUG_CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME}
