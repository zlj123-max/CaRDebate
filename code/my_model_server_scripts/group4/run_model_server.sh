#!/bin/bash

set -u
set -o pipefail


# ==========================================================
# 配置
# ==========================================================

PROJECT_DIR="/code/CaRDebate"

COMMON_CONFIG="${PROJECT_DIR}/config/debate_common_group1.yaml"

UTILS="${PROJECT_DIR}/CaRDebate/utils.py"
PYTHON="/root/miniforge3/envs/CaRDebate/bin/python"


# ==========================================================
# 检查配置
# ==========================================================

if [ ! -f "${COMMON_CONFIG}" ]; then
    echo "ERROR: common config file not found:"
    echo "${COMMON_CONFIG}"
    exit 1
fi

if [ ! -f "${UTILS}" ]; then
    echo "ERROR: utils.py not found:"
    echo "${UTILS}"
    exit 1
fi

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found:"
    echo "${PYTHON}"
    exit 1
fi


# ==========================================================
# 加载公共 YAML 配置
# ==========================================================

CONFIG_EXPORTS=$(
    "${PYTHON}" "${UTILS}" \
        --shell "${COMMON_CONFIG}"
)

CONFIG_EXIT=$?

if [ "${CONFIG_EXIT}" -ne 0 ]; then
    echo "ERROR: failed to load config:"
    echo "${COMMON_CONFIG}"
    exit "${CONFIG_EXIT}"
fi

eval "${CONFIG_EXPORTS}"

# ==========================================================
# Deployment role -> concrete model mapping
#
# utils.py only exposes role-based variables.
# This deployment script maps them to concrete model names.
# ==========================================================

GRANITE_DISPLAY_NAME="${AGENT1_DISPLAY_NAME}"
GRANITE_MODEL_PATH="${AGENT1_MODEL_PATH}"
GRANITE_SERVED_NAME="${AGENT1_SERVED_NAME}"
GPU_GRANITE="${GPU_AGENT1}"
PORT_GRANITE="${PORT_AGENT1}"
MEM_GRANITE="${MEM_AGENT1}"
MAX_LEN_GRANITE="${MAX_LEN_AGENT1}"
GRANITE_URL="${AGENT1_URL}"

HUNYUAN_DISPLAY_NAME="${AGENT2_DISPLAY_NAME}"
HUNYUAN_MODEL_PATH="${AGENT2_MODEL_PATH}"
HUNYUAN_SERVED_NAME="${AGENT2_SERVED_NAME}"
GPU_HUNYUAN="${GPU_AGENT2}"
PORT_HUNYUAN="${PORT_AGENT2}"
MEM_HUNYUAN="${MEM_AGENT2}"
MAX_LEN_HUNYUAN="${MAX_LEN_AGENT2}"
HUNYUAN_URL="${AGENT2_URL}"

FALCON_DISPLAY_NAME="${AGENT3_DISPLAY_NAME}"
FALCON_MODEL_PATH="${AGENT3_MODEL_PATH}"
FALCON_SERVED_NAME="${AGENT3_SERVED_NAME}"
GPU_FALCON="${GPU_AGENT3}"
PORT_FALCON="${PORT_AGENT3}"
MEM_FALCON="${MEM_AGENT3}"
MAX_LEN_FALCON="${MAX_LEN_AGENT3}"
FALCON_URL="${AGENT3_URL}"

GUARD_DISPLAY_NAME="${EVALUATOR_DISPLAY_NAME}"
GUARD_MODEL_PATH="${EVALUATOR_MODEL_PATH}"
GUARD_SERVED_NAME="${EVALUATOR_SERVED_NAME}"
GPU_GUARD="${GPU_EVALUATOR}"
PORT_GUARD="${PORT_EVALUATOR}"
MEM_GUARD="${MEM_EVALUATOR}"
MAX_LEN_GUARD="${MAX_LEN_EVALUATOR}"
LLAMA_GUARD_URL="${EVALUATOR_URL}"



# ==========================================================
# Shared vLLM Model Server
#
# GPU 4:
#   Granite-3.3-8B
#
# GPU 5:
#   Hunyuan-7B
#
# GPU 6:
#   Falcon-H1-7B
#
# GPU 7:
#   Llama-Guard-3-8B
#
# Model parameters are loaded from:
#   debate_common_group4.yaml
#
# This script manages Group4 shared model servers
# and its dedicated OpenAI-compatible Router.
# Algorithm monitors are started by each algorithm run script.
# ==========================================================


# ==========================================================
# Model server directories
# ==========================================================

LOG_DIR="${MODEL_SERVER_LOG}"
PID_DIR="${MODEL_SERVER_PID}"

mkdir -p "${LOG_DIR}"
mkdir -p "${PID_DIR}"


# ==========================================================
# Group4 Router configuration
# ==========================================================

ROUTER_SCRIPT="${PROJECT_DIR}/my_model_server_scripts/group4/model_router.py"
ROUTER_LOG="${LOG_DIR}/model_router_group4.log"
ROUTER_PID_FILE="${PID_DIR}/model_router_group4.pid"


# ==========================================================
# Startup configuration
# ==========================================================

STARTUP_TIMEOUT=300
CHECK_INTERVAL=2


# ==========================================================
# Helper: check server
# ==========================================================

server_ready() {
    local port="$1"

    curl -s \
        --connect-timeout 2 \
        --max-time 5 \
        -H "Authorization: Bearer ${LOCAL_API_KEY}" \
        "http://${HOST}:${port}/v1/models" \
        >/dev/null 2>&1
}


# ==========================================================
# Helper: wait until server is ready
# ==========================================================

wait_for_server() {
    local name="$1"
    local port="$2"
    local pid="$3"
    local logfile="$4"

    local elapsed=0

    echo
    echo "----------------------------------------------------------"
    echo "Waiting for ${name}"
    echo "Port: ${port}"
    echo "PID : ${pid}"
    echo "----------------------------------------------------------"

    while true; do

        if server_ready "${port}"; then
            echo
            echo "[READY] ${name}"
            echo "        http://${HOST}:${port}"
            echo
            return 0
        fi

        if ! kill -0 "${pid}" 2>/dev/null; then
            echo
            echo "[FAILED] ${name} process exited."
            echo
            echo "Last 50 lines of:"
            echo "${logfile}"
            echo "----------------------------------------------------------"
            tail -n 50 "${logfile}"
            echo "----------------------------------------------------------"
            return 1
        fi

        sleep "${CHECK_INTERVAL}"

        elapsed=$((elapsed + CHECK_INTERVAL))

        if [ "${elapsed}" -ge "${STARTUP_TIMEOUT}" ]; then
            echo
            echo "[TIMEOUT] ${name} did not become ready within ${STARTUP_TIMEOUT}s."
            echo
            echo "Last 50 log lines:"
            echo "----------------------------------------------------------"
            tail -n 50 "${logfile}"
            echo "----------------------------------------------------------"
            return 1
        fi

        if (( elapsed % 10 == 0 )); then
            echo "[${name}] still loading... ${elapsed}s"
        fi

    done
}


# ==========================================================
# Helper: start model
# ==========================================================

start_model() {
    local name="$1"
    local model_path="$2"
    local served_name="$3"
    local gpu="$4"
    local port="$5"
    local memory="$6"
    local max_len="$7"
    local logfile="$8"
    local pidfile="$9"
    local structured_output="${10:-false}"

    local extra_args=()
    if [ "${structured_output}" = "true" ]; then
        extra_args+=(
            --structured-outputs-config
            '{"backend":"xgrammar","disable_any_whitespace":true}'
        )
    fi

    echo
    echo "=========================================================="
    echo "Starting ${name}"
    echo "Model              : ${model_path}"
    echo "Served name        : ${served_name}"
    echo "GPU                : ${gpu}"
    echo "Port               : ${port}"
    echo "GPU memory fraction: ${memory}"
    echo "Max model length   : ${max_len}"
    echo "Log                : ${logfile}"
    echo "PID file           : ${pidfile}"
    echo "=========================================================="

    CUDA_VISIBLE_DEVICES="${gpu}" \
    nohup "${VLLM_BIN}" serve \
        "${model_path}" \
        --served-model-name "${served_name}" \
        --host "${HOST}" \
        --port "${port}" \
        --api-key "${LOCAL_API_KEY}" \
        --gpu-memory-utilization "${memory}" \
        --max-model-len "${max_len}" \
        "${extra_args[@]}" \
        > "${logfile}" 2>&1 &

    local pid=$!

    echo "${pid}" > "${pidfile}"

    echo "${name} launched, PID=${pid}"

    wait_for_server \
        "${name}" \
        "${port}" \
        "${pid}" \
        "${logfile}"

    return $?
}


# ==========================================================
# Preliminary checks
# ==========================================================

echo
echo "=========================================================="
echo "Shared model server startup"
echo "=========================================================="
echo "Common config:"
echo "  ${COMMON_CONFIG}"

if [ ! -x "${VLLM_BIN}" ]; then
    echo "ERROR: vLLM executable not found:"
    echo "${VLLM_BIN}"
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not found."
    exit 1
fi

if ! command -v ss >/dev/null 2>&1; then
    echo "ERROR: ss not found."
    exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found."
    exit 1
fi

echo
echo "vLLM:"
"${VLLM_BIN}" --version

echo
echo "GPU status before launch:"

nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader

echo


# ==========================================================
# Print model configuration
# ==========================================================

echo "=========================================================="
echo "Model configuration"
echo "=========================================================="

echo "Granite-3.3-8B:"
echo "  Display: ${GRANITE_DISPLAY_NAME}"
echo "  Model  : ${GRANITE_MODEL_PATH}"
echo "  Served : ${GRANITE_SERVED_NAME}"
echo "  GPU    : ${GPU_GRANITE}"
echo "  Port   : ${PORT_GRANITE}"
echo "  Memory : ${MEM_GRANITE}"
echo "  Length : ${MAX_LEN_GRANITE}"

echo
echo "Hunyuan-7B:"
echo "  Display: ${HUNYUAN_DISPLAY_NAME}"
echo "  Model  : ${HUNYUAN_MODEL_PATH}"
echo "  Served : ${HUNYUAN_SERVED_NAME}"
echo "  GPU    : ${GPU_HUNYUAN}"
echo "  Port   : ${PORT_HUNYUAN}"
echo "  Memory : ${MEM_HUNYUAN}"
echo "  Length : ${MAX_LEN_HUNYUAN}"

echo
echo "Falcon-H1-7B:"
echo "  Display: ${FALCON_DISPLAY_NAME}"
echo "  Model  : ${FALCON_MODEL_PATH}"
echo "  Served : ${FALCON_SERVED_NAME}"
echo "  GPU    : ${GPU_FALCON}"
echo "  Port   : ${PORT_FALCON}"
echo "  Memory : ${MEM_FALCON}"
echo "  Length : ${MAX_LEN_FALCON}"

echo
echo "Llama Guard:"
echo "  Display: ${GUARD_DISPLAY_NAME}"
echo "  Model  : ${GUARD_MODEL_PATH}"
echo "  Served : ${GUARD_SERVED_NAME}"
echo "  GPU    : ${GPU_GUARD}"
echo "  Port   : ${PORT_GUARD}"
echo "  Memory : ${MEM_GUARD}"
echo "  Length : ${MAX_LEN_GUARD}"

echo "=========================================================="
echo


# ==========================================================
# Make sure requested ports are free
# ==========================================================

for PORT in \
    "${PORT_GRANITE}" \
    "${PORT_HUNYUAN}" \
    "${PORT_FALCON}" \
    "${PORT_GUARD}"
do
    if ss -lnt | grep -q ":${PORT} "; then
        echo "ERROR: port ${PORT} is already in use."
        echo

        ss -lntp | grep ":${PORT} " || true

        echo
        echo "Stop the old vLLM service first."
        exit 1
    fi
done


# ==========================================================
# Clear stale PID files
# ==========================================================

rm -f \
    "${PID_DIR}/granite.pid" \
    "${PID_DIR}/hunyuan.pid" \
    "${PID_DIR}/falcon.pid" \
    "${PID_DIR}/llama_guard.pid"


# ==========================================================
# GPU 0 - Mistral
# ==========================================================

start_model \
    "${GRANITE_DISPLAY_NAME}" \
    "${GRANITE_MODEL_PATH}" \
    "${GRANITE_SERVED_NAME}" \
    "${GPU_GRANITE}" \
    "${PORT_GRANITE}" \
    "${MEM_GRANITE}" \
    "${MAX_LEN_GRANITE}" \
    "${LOG_DIR}/granite.log" \
    "${PID_DIR}/granite.pid" \
    true

if [ $? -ne 0 ]; then
    echo "Granite-3.3-8B failed to start."
    exit 1
fi


# ==========================================================
# GPU 1 - Llama
# ==========================================================

start_model \
    "${HUNYUAN_DISPLAY_NAME}" \
    "${HUNYUAN_MODEL_PATH}" \
    "${HUNYUAN_SERVED_NAME}" \
    "${GPU_HUNYUAN}" \
    "${PORT_HUNYUAN}" \
    "${MEM_HUNYUAN}" \
    "${MAX_LEN_HUNYUAN}" \
    "${LOG_DIR}/hunyuan.log" \
    "${PID_DIR}/hunyuan.pid" \
    true

if [ $? -ne 0 ]; then
    echo "Hunyuan-7B failed to start."
    exit 1
fi


# ==========================================================
# GPU 2 - Phi
# ==========================================================

start_model \
    "${FALCON_DISPLAY_NAME}" \
    "${FALCON_MODEL_PATH}" \
    "${FALCON_SERVED_NAME}" \
    "${GPU_FALCON}" \
    "${PORT_FALCON}" \
    "${MEM_FALCON}" \
    "${MAX_LEN_FALCON}" \
    "${LOG_DIR}/falcon.log" \
    "${PID_DIR}/falcon.pid" \
    true

if [ $? -ne 0 ]; then
    echo "Falcon-H1-7B failed to start."
    exit 1
fi


# ==========================================================
# GPU 7 - Llama Guard
# ==========================================================

start_model \
    "${GUARD_DISPLAY_NAME}" \
    "${GUARD_MODEL_PATH}" \
    "${GUARD_SERVED_NAME}" \
    "${GPU_GUARD}" \
    "${PORT_GUARD}" \
    "${MEM_GUARD}" \
    "${MAX_LEN_GUARD}" \
    "${LOG_DIR}/llama_guard.log" \
    "${PID_DIR}/llama_guard.pid" \
    false

if [ $? -ne 0 ]; then
    echo "Llama Guard failed to start."
    exit 1
fi


# ==========================================================
# Final health check
# ==========================================================

echo
echo "=========================================================="
echo "Final model server health check"
echo "=========================================================="

ALL_OK=true

check_final() {
    local name="$1"
    local port="$2"

    if server_ready "${port}"; then
        printf "%-22s : OK   port=%s\n" \
            "${name}" \
            "${port}"
    else
        printf "%-22s : FAILED   port=%s\n" \
            "${name}" \
            "${port}"

        ALL_OK=false
    fi
}

check_final \
    "${GRANITE_DISPLAY_NAME}" \
    "${PORT_GRANITE}"

check_final \
    "${HUNYUAN_DISPLAY_NAME}" \
    "${PORT_HUNYUAN}"

check_final \
    "${FALCON_DISPLAY_NAME}" \
    "${PORT_FALCON}"

check_final \
    "${GUARD_DISPLAY_NAME}" \
    "${PORT_GUARD}"

if [ "${ALL_OK}" != true ]; then
    echo
    echo "One or more model servers failed."
    exit 1
fi


# ==========================================================
# Always redeploy Group4 OpenAI Router
#
# Policy:
#   Every successful model deployment rebuilds the Router.
#   Existing process on ROUTER_PORT is stopped directly.
#   Existing Router is never reused.
# ==========================================================

echo
echo "=========================================================="
echo "Redeploying Group4 OpenAI Router"
echo "=========================================================="
echo "Router URL : ${ROUTER_URL}"
echo "Router port: ${ROUTER_PORT}"
echo "Router log : ${ROUTER_LOG}"
echo "=========================================================="

if [ ! -f "${ROUTER_SCRIPT}" ]; then
    echo "ERROR: Router script not found:"
    echo "${ROUTER_SCRIPT}"
    exit 1
fi

if [ ! -x "${VLLM_PYTHON}" ]; then
    echo "ERROR: vLLM Python not found:"
    echo "${VLLM_PYTHON}"
    exit 1
fi

if [ -z "${REMOTE_OPENAI_API_KEY:-}" ]; then
    echo "ERROR: REMOTE_OPENAI_API_KEY is not set."
    echo
    echo "Please export the remote OpenAI-compatible API key before deployment."
    exit 1
fi


# ==========================================================
# Stop anything currently listening on Group4 Router port
# ==========================================================

EXISTING_ROUTER_PID=$(
    ss -lntp 2>/dev/null \
        | grep ":${ROUTER_PORT} " \
        | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' \
        | head -1
)

if [ -n "${EXISTING_ROUTER_PID}" ]; then
    echo
    echo "Stopping existing process on Router port ${ROUTER_PORT}:"
    echo "  PID=${EXISTING_ROUTER_PID}"

    kill "${EXISTING_ROUTER_PID}" \
        2>/dev/null || true

    for i in $(seq 1 20); do
        if ! ss -lnt \
            | grep -q ":${ROUTER_PORT} "
        then
            break
        fi

        sleep 1
    done

    if ss -lnt \
        | grep -q ":${ROUTER_PORT} "
    then
        echo
        echo "Router port ${ROUTER_PORT} is still occupied."
        echo "Force stopping existing process..."

        kill -9 "${EXISTING_ROUTER_PID}" \
            2>/dev/null || true

        sleep 1
    fi
fi

if ss -lnt \
    | grep -q ":${ROUTER_PORT} "
then
    echo
    echo "ERROR: Router port ${ROUTER_PORT} is still occupied."
    ss -lntp \
        | grep ":${ROUTER_PORT} " \
        || true
    exit 1
fi


# ==========================================================
# Clear stale Group4 Router PID file
# ==========================================================

rm -f "${ROUTER_PID_FILE}"


# ==========================================================
# Always start a fresh Router from CURRENT Group4 config
# ==========================================================

echo
echo "Starting fresh Group4 Router..."

REMOTE_OPENAI_API_KEY="${REMOTE_OPENAI_API_KEY}" \
REMOTE_OPENAI_BASE_URL="${REMOTE_OPENAI_BASE_URL}" \
LOCAL_MODEL_ROUTES_JSON="${LOCAL_MODEL_ROUTES_JSON}" \
nohup "${VLLM_PYTHON}" -m uvicorn \
    model_router:app \
    --app-dir "${PROJECT_DIR}/my_model_server_scripts/group4" \
    --host "${HOST}" \
    --port "${ROUTER_PORT}" \
    > "${ROUTER_LOG}" 2>&1 &

ROUTER_PID=$!

echo "${ROUTER_PID}" \
    > "${ROUTER_PID_FILE}"

echo
echo "Router launched:"
echo "  PID : ${ROUTER_PID}"
echo "  Port: ${ROUTER_PORT}"
echo "  Log : ${ROUTER_LOG}"


# ==========================================================
# Wait for fresh Group4 Router
# ==========================================================

ROUTER_READY=false

for i in $(seq 1 30); do
    if curl -sf \
        --connect-timeout 2 \
        --max-time 5 \
        "${ROUTER_URL}/v1/models" \
        >/dev/null
    then
        ROUTER_READY=true
        break
    fi

    if ! kill -0 "${ROUTER_PID}" \
        2>/dev/null
    then
        echo
        echo "ERROR: Group4 Router process exited."
        echo
        echo "Last 100 Router log lines:"
        echo "----------------------------------------------------------"
        tail -n 100 "${ROUTER_LOG}" \
            2>/dev/null || true
        echo "----------------------------------------------------------"
        exit 1
    fi

    sleep 1
done

if [ "${ROUTER_READY}" != true ]; then
    echo
    echo "ERROR: Group4 Router did not become ready."
    echo
    echo "Last 100 Router log lines:"
    echo "----------------------------------------------------------"
    tail -n 100 "${ROUTER_LOG}" \
        2>/dev/null || true
    echo "----------------------------------------------------------"
    exit 1
fi

echo
echo "[READY] Fresh Group4 OpenAI Router"
echo "        ${ROUTER_URL}"
echo "        PID=${ROUTER_PID}"
echo


# ==========================================================
# GPU status
# ==========================================================

echo
echo "=========================================================="
echo "GPU status after launch"
echo "=========================================================="

nvidia-smi \
    --query-gpu=index,name,memory.total,memory.used,memory.free \
    --format=csv,noheader


# ==========================================================
# Summary
# ==========================================================

echo
echo "=========================================================="
echo "ALL MODEL SERVICES READY"
echo "=========================================================="

echo
echo "GPU ${GPU_GRANITE}:"
echo "  ${GRANITE_DISPLAY_NAME}"
echo "    ${GRANITE_URL}"
echo "    gpu-memory-utilization=${MEM_GRANITE}"
echo "    max-model-len=${MAX_LEN_GRANITE}"

echo
echo "GPU ${GPU_HUNYUAN}:"
echo "  ${HUNYUAN_DISPLAY_NAME}"
echo "    ${HUNYUAN_URL}"
echo "    gpu-memory-utilization=${MEM_HUNYUAN}"
echo "    max-model-len=${MAX_LEN_HUNYUAN}"

echo
echo "GPU ${GPU_FALCON}:"
echo "  ${FALCON_DISPLAY_NAME}"
echo "    ${FALCON_URL}"
echo "    gpu-memory-utilization=${MEM_FALCON}"
echo "    max-model-len=${MAX_LEN_FALCON}"

echo
echo "GPU ${GPU_GUARD}:"
echo "  ${GUARD_DISPLAY_NAME}"
echo "    ${LLAMA_GUARD_URL}"
echo "    gpu-memory-utilization=${MEM_GUARD}"
echo "    max-model-len=${MAX_LEN_GUARD}"

echo
echo "Group4 Router:"
echo "  URL : ${ROUTER_URL}"
echo "  PID : ${ROUTER_PID}"
echo "  Log : ${ROUTER_LOG}"

echo
echo "Logs:"
echo "  ${LOG_DIR}"

echo
echo "PIDs:"
echo "  ${PID_DIR}"

echo
echo "=========================================================="
echo "Group4 model services and freshly redeployed Router are ready."
echo
echo "Model output monitor is NOT started here."
echo "Each algorithm run script starts its own monitor."
echo "=========================================================="