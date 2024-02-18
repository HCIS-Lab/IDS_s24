export CARLA_ROOT=$1
export HOMEWORK_DIR=$2
export MODEL_NAME=$3

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export LEADERBOARD_ROOT=${HOMEWORK_DIR}/longest6
export SCENARIO_RUNNER_ROOT=${LEADERBOARD_ROOT}/scenario_runner
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export SCENARIOS=${LEADERBOARD_ROOT}/data/scenarios/eval_scenarios.json
export ROUTES=${LEADERBOARD_ROOT}/data/longest6_tiny.xml
export REPETITIONS=1
export CHALLENGE_TRACK_CODENAME=MAP
export CHECKPOINT_ENDPOINT=${HOMEWORK_DIR}/results/longest6/${MODEL_NAME}/longest6.json
export TEAM_AGENT=${HOMEWORK_DIR}/agents/${MODEL_NAME}/longest6_agent.py
export TEAM_CONFIG=${HOMEWORK_DIR}/agents/checkpoints/${MODEL_NAME}
export DEBUG_CHALLENGE=0
export RESUME=1
export DATAGEN=0
export SAVE_PATH=${HOMEWORK_DIR}/results/longest6/${MODEL_NAME}
export UNCERTAINTY_THRESHOLD=0.33
export DIRECT=1
export CARLA_PORT=2000
export TM_PORT=5000

export CUDA_VISIBLE_DEVICES=0

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--resume=${RESUME} \
--timeout=600 \
--port=${CARLA_PORT} \
--trafficManagerPort=${TM_PORT}

