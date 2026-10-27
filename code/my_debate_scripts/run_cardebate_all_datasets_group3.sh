#!/bin/bash

set -u
set -o pipefail


# ============================================================
# 配置
# ============================================================

PROJECT_DIR="/code/CaRDebate"

COMMON_CONFIG="${PROJECT_DIR}/config/debate_common_group1.yaml"
ALGO_CONFIG="${PROJECT_DIR}/config/algo/cardebate.yaml"

UTILS="${PROJECT_DIR}/CaRDebate/utils.py"
PYTHON="/root/miniforge3/envs/CaRDebate/bin/python"


# ============================================================
# 依次运行的数据集
# ============================================================

DATASET_NAMES=(
    "harmbench"
    "beavertails_evaluation"
    "cosafe"
    "aegis2"
    "wildjailbreak"
)

DATASET_PATHS=(
    "/datasets/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    "/datasets/PKU-Alignment/BeaverTails-Evaluation/v1/beavertails_evaluation.csv"
    "/datasets/CoSafe-Dataset/cosafe.csv"
    "/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/aegis2_core.csv"
    "/datasets/allenai/wildjailbreak/train/train.csv"
)


# ============================================================
# 检查配置
# ============================================================

if [ ! -f "${COMMON_CONFIG}" ]; then
    echo "ERROR: common config file not found:"
    echo "${COMMON_CONFIG}"
    exit 1
fi

if [ ! -f "${ALGO_CONFIG}" ]; then
    echo "ERROR: algorithm config file not found:"
    echo "${ALGO_CONFIG}"
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

if [ "${#DATASET_NAMES[@]}" -ne "${#DATASET_PATHS[@]}" ]; then
    echo "ERROR: DATASET_NAMES and DATASET_PATHS length mismatch"
    exit 1
fi


# ============================================================
# 加载 YAML 配置
# ============================================================

CONFIG_EXPORTS=$(
    "${PYTHON}" "${UTILS}" \
        --shell \
        "${COMMON_CONFIG}" \
        "${ALGO_CONFIG}"
)

CONFIG_EXIT=$?

if [ "${CONFIG_EXIT}" -ne 0 ]; then
    echo "ERROR: failed to load config:"
    echo "${COMMON_CONFIG}"
    echo "${ALGO_CONFIG}"
    exit "${CONFIG_EXIT}"
fi

eval "${CONFIG_EXPORTS}"


# ============================================================
# 本批实验时间戳
# ============================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

if [ -z "${ANSWER_MAX_WORDS:-}" ]; then
    echo "ERROR: ANSWER_MAX_WORDS was not exported from ${COMMON_CONFIG}"
    exit 1
fi

if [ -z "${FORMAT_REPAIR_MAX_WORDS:-}" ]; then
    echo "ERROR: FORMAT_REPAIR_MAX_WORDS was not exported from ${COMMON_CONFIG}"
    exit 1
fi

if ! [[ "${ANSWER_MAX_WORDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ANSWER_MAX_WORDS must be a positive integer: ${ANSWER_MAX_WORDS}"
    exit 1
fi

if ! [[ "${FORMAT_REPAIR_MAX_WORDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: FORMAT_REPAIR_MAX_WORDS must be a positive integer: ${FORMAT_REPAIR_MAX_WORDS}"
    exit 1
fi

mkdir -p "${MODEL_SERVER_LOG}"
mkdir -p "${MODEL_SERVER_PID}"

ROUTER_LOG="${MODEL_SERVER_LOG}/model_router.log"

MONITOR_SCRIPT="${PROJECT_DIR}/my_model_server_scripts/group3/run_check_model_output.sh"


# ============================================================
# GPU
# ============================================================

# CaRDebate 本身不使用 GPU。
# 四个本地 vLLM server 已独立管理 GPU。
export CUDA_VISIBLE_DEVICES=""


# ============================================================
# 外部 OpenAI-compatible API
# ============================================================

# 不要把真实 Key 写进 YAML。
#
# 启动前：
#
# export REMOTE_OPENAI_API_KEY='你的key'

if [ -z "${REMOTE_OPENAI_API_KEY:-}" ]; then
    echo "ERROR: REMOTE_OPENAI_API_KEY 未设置"
    echo
    echo "请先执行："
    echo "export REMOTE_OPENAI_API_KEY='你的key'"
    exit 1
fi

export REMOTE_OPENAI_API_KEY


# ============================================================
# 环境配置
# ============================================================

export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${CUDA_HOME}/lib64"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export OPENAI_API_KEY
export OPENAI_BASE_URL
export REMOTE_OPENAI_BASE_URL
export LOCAL_MODEL_ROUTES_JSON


# ============================================================
# 脚本路径
# ============================================================

ROUTER_SCRIPT="${PROJECT_DIR}/my_model_server_scripts/group3/model_router.py"


# ============================================================
# 日志函数
# ============================================================

LOG_FILE=""

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" \
        | tee -a "${LOG_FILE}"
}


# ============================================================
# Monitor 清理函数
# ============================================================

MODEL_MONITOR_PID=""
MONITOR_PID_FILE=""

stop_model_monitor() {
    if [ -z "${MODEL_MONITOR_PID}" ]; then
        return
    fi

    if kill -0 "${MODEL_MONITOR_PID}" 2>/dev/null; then
        if command -v setsid >/dev/null 2>&1; then
            kill -TERM -- "-${MODEL_MONITOR_PID}" \
                2>/dev/null || true
        else
            pkill -P "${MODEL_MONITOR_PID}" \
                2>/dev/null || true

            kill "${MODEL_MONITOR_PID}" \
                2>/dev/null || true
        fi
    fi

    wait "${MODEL_MONITOR_PID}" \
        2>/dev/null || true

    if [ -n "${MONITOR_PID_FILE}" ]; then
        rm -f "${MONITOR_PID_FILE}"
    fi

    MODEL_MONITOR_PID=""
}

trap stop_model_monitor EXIT
trap 'exit 130' INT
trap 'exit 143' TERM


# ============================================================
# 基础检查
# ============================================================

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "ERROR: 找不到 CaRDebate Python:"
    echo "${PYTHON_BIN}"
    exit 1
fi

if [ ! -x "${VLLM_PYTHON}" ]; then
    echo "ERROR: 找不到 vLLM Python:"
    echo "${VLLM_PYTHON}"
    exit 1
fi

if [ ! -d "${PROJECT_DIR}" ]; then
    echo "ERROR: 项目目录不存在:"
    echo "${PROJECT_DIR}"
    exit 1
fi

if [ ! -f "${PROJECT_DIR}/main_cardebate.py" ]; then
    echo "ERROR: main_cardebate.py 不存在"
    exit 1
fi

if [ ! -f "${ROUTER_SCRIPT}" ]; then
    echo "ERROR: Router 不存在:"
    echo "${ROUTER_SCRIPT}"
    exit 1
fi

if [ ! -f "${MONITOR_SCRIPT}" ]; then
    echo "ERROR: Monitor 不存在:"
    echo "${MONITOR_SCRIPT}"
    exit 1
fi

for i in "${!DATASET_NAMES[@]}"; do
    if [ ! -f "${DATASET_PATHS[$i]}" ]; then
        echo "ERROR: 数据集不存在:"
        echo "${DATASET_NAMES[$i]}: ${DATASET_PATHS[$i]}"
        exit 1
    fi
done


# ============================================================
# 检查四个 vLLM 服务
# ============================================================

check_vllm() {
    NAME="$1"
    PORT="$2"

    echo "Checking ${NAME} on port ${PORT} ..."

    if ! curl -sf \
        --connect-timeout 3 \
        --max-time 10 \
        -H "Authorization: Bearer ${LOCAL_API_KEY}" \
        "http://${HOST}:${PORT}/v1/models" \
        >/dev/null
    then
        echo "ERROR: ${NAME} vLLM server 不可用"
        echo "Port: ${PORT}"
        exit 1
    fi

    echo "${NAME}: OK"
}


echo
echo "============================================================"
echo "Checking vLLM servers"
echo "============================================================"

check_vllm "Agent1 (${AGENT1_DISPLAY_NAME})" "${PORT_AGENT1}"
check_vllm "Agent2 (${AGENT2_DISPLAY_NAME})" "${PORT_AGENT2}"
check_vllm "Agent3 (${AGENT3_DISPLAY_NAME})" "${PORT_AGENT3}"
check_vllm "Evaluator (${EVALUATOR_DISPLAY_NAME})" "${PORT_EVALUATOR}"


# ============================================================
# 启动本地 OpenAI Router
# ============================================================

if curl -sf \
    --connect-timeout 2 \
    "${ROUTER_URL}/v1/models" \
    >/dev/null 2>&1
then
    echo "OpenAI Router already running."
else
    echo "Starting OpenAI Router..."

    REMOTE_OPENAI_API_KEY="${REMOTE_OPENAI_API_KEY}" \
    REMOTE_OPENAI_BASE_URL="${REMOTE_OPENAI_BASE_URL}" \
    LOCAL_MODEL_ROUTES_JSON="${LOCAL_MODEL_ROUTES_JSON}" \
    nohup "${VLLM_PYTHON}" -m uvicorn \
        model_router:app \
        --app-dir "${PROJECT_DIR}/my_model_server_scripts/group3" \
        --host "${HOST}" \
        --port "${ROUTER_PORT}" \
        > "${ROUTER_LOG}" 2>&1 &

    ROUTER_PID=$!

    echo "Router PID: ${ROUTER_PID}"

    for i in $(seq 1 30); do
        if curl -sf \
            "${ROUTER_URL}/v1/models" \
            >/dev/null 2>&1
        then
            echo "OpenAI Router: READY"
            break
        fi

        if ! kill -0 "${ROUTER_PID}" 2>/dev/null; then
            echo "ERROR: OpenAI Router 启动失败"
            tail -n 50 "${ROUTER_LOG}"
            exit 1
        fi

        sleep 1

        if [ "${i}" -eq 30 ]; then
            echo "ERROR: OpenAI Router 启动超时"
            tail -n 50 "${ROUTER_LOG}"
            exit 1
        fi
    done
fi


# ============================================================
# 进入项目目录
# ============================================================

cd "${PROJECT_DIR}" || exit 1


# ============================================================
# 为单个数据集生成独立 common YAML
# ============================================================

create_dataset_common_config() {
    local dataset_name="$1"
    local dataset_path="$2"
    local output_config="$3"

    BASE_COMMON_CONFIG="${COMMON_CONFIG}" \
    DATASET_OVERRIDE_NAME="${dataset_name}" \
    DATASET_OVERRIDE_PATH="${dataset_path}" \
    OUTPUT_COMMON_CONFIG="${output_config}" \
    "${PYTHON}" - <<'PY'
import os
from pathlib import Path
import yaml

base = Path(os.environ["BASE_COMMON_CONFIG"])
out = Path(os.environ["OUTPUT_COMMON_CONFIG"])

with base.open("r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

if not isinstance(config, dict):
    raise RuntimeError("Common YAML root must be a mapping")

config["dataset"] = {
    "name": os.environ["DATASET_OVERRIDE_NAME"],
    "path": os.environ["DATASET_OVERRIDE_PATH"],
}

out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(
        config,
        f,
        allow_unicode=True,
        sort_keys=False,
    )
PY
}


# ============================================================
# 单个数据集实验
# ============================================================

run_one_dataset() {
    local requested_dataset_name="$1"
    local requested_dataset_path="$2"

    # ========================================================
    # 本数据集独立实验目录
    # ========================================================

    RUN_ROOT="${DATA_ROOT}/${ALGO_NAME}/${requested_dataset_name}_response${ANSWER_MAX_WORDS}_group3_${TIMESTAMP}"

    LOG_ROOT="${RUN_ROOT}/logs"
    CHROMA_DB_ROOT="${RUN_ROOT}/chroma_db"
    CHECKPOINT_DIR="${RUN_ROOT}/checkpoints"
    RESULTS_DIR="${RUN_ROOT}/results"
    MODEL_OUTPUT_DIR="${RUN_ROOT}/model_outputs"
    CONFIG_ROOT="${RUN_ROOT}/config"

    mkdir -p "${LOG_ROOT}"
    mkdir -p "${CHROMA_DB_ROOT}"
    mkdir -p "${CHECKPOINT_DIR}"
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${MODEL_OUTPUT_DIR}"
    mkdir -p "${CONFIG_ROOT}"

    LOG_FILE="${LOG_ROOT}/debate.log"

    MONITOR_LOG="${LOG_ROOT}/model_output_monitor.log"
    MONITOR_PID_FILE="${RUN_ROOT}/model_output_monitor.pid"

    DATASET_COMMON_CONFIG="${CONFIG_ROOT}/debate_common_group3.yaml"

    create_dataset_common_config \
        "${requested_dataset_name}" \
        "${requested_dataset_path}" \
        "${DATASET_COMMON_CONFIG}"


    # ========================================================
    # 针对本数据集重新加载 YAML 配置
    # ========================================================

    DATASET_CONFIG_EXPORTS=$(
        "${PYTHON}" "${UTILS}" \
            --shell \
            "${DATASET_COMMON_CONFIG}" \
            "${ALGO_CONFIG}"
    )

    DATASET_CONFIG_EXIT=$?

    if [ "${DATASET_CONFIG_EXIT}" -ne 0 ]; then
        echo "ERROR: failed to load dataset config:"
        echo "${DATASET_COMMON_CONFIG}"
        return "${DATASET_CONFIG_EXIT}"
    fi

    eval "${DATASET_CONFIG_EXPORTS}"

    export CHROMA_DB_ROOT
    export CHECKPOINT_DIR
    export RESULTS_DIR


    # ========================================================
    # 打印运行环境
    # ========================================================

    log "============================================================"
    log "CaRDebate started - vLLM server mode"
    log "Algorithm          : ${ALGO_NAME}"
    log "Common config      : ${DATASET_COMMON_CONFIG}"
    log "Algorithm config   : ${ALGO_CONFIG}"
    log "Run root           : ${RUN_ROOT}"
    log "Project            : ${PROJECT_DIR}"
    log "Python             : ${PYTHON_BIN}"
    log "Log                : ${LOG_FILE}"
    log "Model output       : ${MODEL_OUTPUT_DIR}"
    log "Chroma DB          : ${CHROMA_DB_ROOT}"
    log "Checkpoint         : ${CHECKPOINT_DIR}"
    log "Results            : ${RESULTS_DIR}"
    log "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    log "Agent1             : ${AGENT1_URL}"
    log "Agent2             : ${AGENT2_URL}"
    log "Agent3             : ${AGENT3_URL}"
    log "Evaluator          : ${EVALUATOR_URL}"
    log "OpenAI Router      : ${ROUTER_URL}"
    log "Dataset name       : ${DATASET_NAME}"
    log "Dataset            : ${DATASET_PATH}"
    log "Debate rounds      : ${DEBATE_ROUNDS}"
    log "============================================================"


    # ========================================================
    # 环境预检查
    # ========================================================

    log "Running Python/SQLite/Chroma preflight check..."

    "${PYTHON_BIN}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import os
import sys
import sqlite3

print("Python:", sys.executable)
print("Version:", sys.version)
print("SQLite:", sqlite3.sqlite_version)

import yaml
print("PyYAML:", yaml.__version__)

import chromadb
print("ChromaDB:", chromadb.__version__)

from langchain_chroma import Chroma
print("langchain_chroma: OK")

print(
    "CUDA_VISIBLE_DEVICES:",
    repr(os.environ.get("CUDA_VISIBLE_DEVICES"))
)

print(
    "OPENAI_BASE_URL:",
    os.environ.get("OPENAI_BASE_URL")
)

print(
    "CHROMA_DB_ROOT:",
    os.environ.get("CHROMA_DB_ROOT")
)

print(
    "CHECKPOINT_DIR:",
    os.environ.get("CHECKPOINT_DIR")
)

print(
    "RESULTS_DIR:",
    os.environ.get("RESULTS_DIR")
)

print("Preflight check: OK")
PY

    PREFLIGHT_EXIT=${PIPESTATUS[0]}

    if [ "${PREFLIGHT_EXIT}" -ne 0 ]; then
        log "ERROR: Python/Chroma 环境检查失败。"
        return "${PREFLIGHT_EXIT}"
    fi


    # ========================================================
    # GPU 信息
    # ========================================================

    if command -v nvidia-smi >/dev/null 2>&1; then
        log "GPU status:"

        nvidia-smi \
            --query-gpu=index,name,memory.total,memory.used \
            --format=csv,noheader \
            2>&1 | tee -a "${LOG_FILE}"
    fi


    # ========================================================
    # 创建本次 debate.log
    # ========================================================

    touch "${LOG_FILE}"


    # ========================================================
    # 启动本次算法模型输出监控
    # ========================================================

    if command -v setsid >/dev/null 2>&1; then
        DEBATE_LOG_DIR="${LOG_ROOT}" \
        OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
        nohup setsid bash "${MONITOR_SCRIPT}" \
            > "${MONITOR_LOG}" 2>&1 &
    else
        DEBATE_LOG_DIR="${LOG_ROOT}" \
        OUTPUT_DIR="${MODEL_OUTPUT_DIR}" \
        nohup bash "${MONITOR_SCRIPT}" \
            > "${MONITOR_LOG}" 2>&1 &
    fi

    MODEL_MONITOR_PID=$!

    echo "${MODEL_MONITOR_PID}" \
        > "${MONITOR_PID_FILE}"

    sleep 1

    if ! kill -0 "${MODEL_MONITOR_PID}" 2>/dev/null; then
        echo "ERROR: model output monitor failed to start."
        echo "Monitor log:"
        tail -n 100 "${MONITOR_LOG}" 2>/dev/null || true
        MODEL_MONITOR_PID=""
        return 1
    fi

    log "Model output monitor started."
    log "Monitor PID        : ${MODEL_MONITOR_PID}"
    log "Monitor PID file   : ${MONITOR_PID_FILE}"
    log "Monitor log        : ${MONITOR_LOG}"


    # ========================================================
    # CaRDebate 配置检查
    # ========================================================

    if [ ! -f "${DATASET_COMMON_CONFIG}" ]; then
        log "ERROR: Common config does not exist: ${DATASET_COMMON_CONFIG}"
        stop_model_monitor
        return 1
    fi

    if [ ! -f "${ALGO_CONFIG}" ]; then
        log "ERROR: Algorithm config does not exist: ${ALGO_CONFIG}"
        stop_model_monitor
        return 1
    fi

    if [ ! -d "${CHECKPOINT_DIR}" ]; then
        log "ERROR: Checkpoint directory does not exist: ${CHECKPOINT_DIR}"
        stop_model_monitor
        return 1
    fi

    if [ ! -d "${MODEL_OUTPUT_DIR}" ]; then
        log "ERROR: Model output directory does not exist: ${MODEL_OUTPUT_DIR}"
        stop_model_monitor
        return 1
    fi


    # ========================================================
    # 启动 CaRDebate
    #
    # 实验相关参数统一由 YAML 提供：
    #
    #   本数据集独立 common YAML
    #           +
    #   config/algo/cardebate.yaml
    #
    # shell 只负责提供本次运行相关路径：
    #   output_file
    #   checkpoint_dir
    #   model_output_dir
    #
    # 不再从 shell 重复传递：
    #   models
    #   evaluator
    #   feedback_generator
    #   datasets
    #   debate_rounds
    #   textual_memory_index
    #   memory / reliability / evaluator / generation 等实验超参数
    # ========================================================

    log "Starting CaRDebate..."
    log "Common config      : ${DATASET_COMMON_CONFIG}"
    log "Algorithm config   : ${ALGO_CONFIG}"
    log "Output file        : ${LOG_FILE}"
    log "Checkpoint dir     : ${CHECKPOINT_DIR}"
    log "Model output dir   : ${MODEL_OUTPUT_DIR}"

    "${PYTHON_BIN}" main_cardebate.py \
        --algorithm-config "${ALGO_CONFIG}" \
        --common-config "${DATASET_COMMON_CONFIG}" \
        --output-file "${LOG_FILE}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --model-output-dir "${MODEL_OUTPUT_DIR}" \
        2>&1 | tee -a "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}


    # ========================================================
    # 结束本次算法 monitor
    # ========================================================

    stop_model_monitor


    # ========================================================
    # 结束信息
    # ========================================================

    log "============================================================"

    if [ "${EXIT_CODE}" -eq 0 ]; then
        log "CaRDebate finished successfully."
    else
        log "CaRDebate failed."
    fi

    log "Exit code          : ${EXIT_CODE}"
    log "Run root           : ${RUN_ROOT}"
    log "Log                : ${LOG_FILE}"
    log "Model output       : ${MODEL_OUTPUT_DIR}"
    log "Chroma DB          : ${CHROMA_DB_ROOT}"
    log "Checkpoint         : ${CHECKPOINT_DIR}"
    log "Results            : ${RESULTS_DIR}"
    log "============================================================"

    return "${EXIT_CODE}"
}


# ============================================================
# 依次运行全部数据集
# ============================================================

TOTAL_DATASETS=${#DATASET_NAMES[@]}

for i in "${!DATASET_NAMES[@]}"; do
    CURRENT_INDEX=$((i + 1))
    CURRENT_DATASET_NAME="${DATASET_NAMES[$i]}"
    CURRENT_DATASET_PATH="${DATASET_PATHS[$i]}"

    echo
    echo "============================================================"
    echo "Dataset ${CURRENT_INDEX}/${TOTAL_DATASETS}: ${CURRENT_DATASET_NAME}"
    echo "============================================================"
    echo "${CURRENT_DATASET_PATH}"
    echo "============================================================"

    run_one_dataset \
        "${CURRENT_DATASET_NAME}" \
        "${CURRENT_DATASET_PATH}"

    DATASET_EXIT_CODE=$?

    if [ "${DATASET_EXIT_CODE}" -ne 0 ]; then
        echo
        echo "============================================================"
        echo "ERROR: ${CURRENT_DATASET_NAME} failed."
        echo "Later datasets will not be started."
        echo "Exit code: ${DATASET_EXIT_CODE}"
        echo "============================================================"
        exit "${DATASET_EXIT_CODE}"
    fi

done


echo
echo "============================================================"
echo "All ${TOTAL_DATASETS} datasets finished successfully."
echo "Batch timestamp: ${TIMESTAMP}"
echo "============================================================"

exit 0
