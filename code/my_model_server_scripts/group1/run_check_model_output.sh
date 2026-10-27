#!/bin/bash

set -u


# ============================================================
# 配置
# ============================================================

PROJECT_DIR="/code/CaRDebate"

COMMON_CONFIG="${PROJECT_DIR}/config/debate_common_group1.yaml"

UTILS="${PROJECT_DIR}/CaRDebate/utils.py"
PYTHON="/root/miniforge3/envs/CaRDebate/bin/python"


# ============================================================
# 检查配置
# ============================================================

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


# ============================================================
# 加载公共 YAML 配置
# ============================================================

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


# ============================================================
# Deployment role -> concrete model mapping
# ============================================================

MISTRAL_OUTPUT_FILE="${AGENT1_OUTPUT_FILE}"
LLAMA_OUTPUT_FILE="${AGENT2_OUTPUT_FILE}"
PHI_OUTPUT_FILE="${AGENT3_OUTPUT_FILE}"
GUARD_OUTPUT_FILE="${EVALUATOR_OUTPUT_FILE}"

MISTRAL_DISPLAY_NAME="${AGENT1_DISPLAY_NAME}"
LLAMA_DISPLAY_NAME="${AGENT2_DISPLAY_NAME}"
PHI_DISPLAY_NAME="${AGENT3_DISPLAY_NAME}"
GUARD_DISPLAY_NAME="${EVALUATOR_DISPLAY_NAME}"


# ============================================================
# 当前算法运行路径
# ============================================================

if [ -z "${DEBATE_LOG_DIR:-}" ]; then
    echo "ERROR: DEBATE_LOG_DIR is not set."
    echo
    echo "This monitor must be started by an algorithm run script."
    exit 1
fi

if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "ERROR: OUTPUT_DIR is not set."
    echo
    echo "This monitor must be started by an algorithm run script."
    exit 1
fi

mkdir -p "${DEBATE_LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

DEBATE_LOG="${DEBATE_LOG_DIR}/debate.log"

PYTHON="${PYTHON_BIN}"


# ============================================================
# 导出给 Python monitor
# ============================================================

export OUTPUT_DIR

export MISTRAL_OUTPUT_FILE
export LLAMA_OUTPUT_FILE
export PHI_OUTPUT_FILE
export GUARD_OUTPUT_FILE
export FEEDBACK_OUTPUT_FILE

export MISTRAL_DISPLAY_NAME
export LLAMA_DISPLAY_NAME
export PHI_DISPLAY_NAME
export GUARD_DISPLAY_NAME


# ============================================================
# 清理函数
# ============================================================

cleanup() {
    trap - EXIT INT TERM

    CHILD_PIDS=$(
        jobs -pr 2>/dev/null || true
    )

    if [ -n "${CHILD_PIDS}" ]; then
        kill ${CHILD_PIDS} \
            2>/dev/null || true
    fi

    wait \
        2>/dev/null || true
}

trap cleanup EXIT INT TERM


# ============================================================
# 启动信息
# ============================================================

echo "============================================================"
echo "Model output monitor"
echo "============================================================"

echo "Common config:"
echo "  ${COMMON_CONFIG}"

echo
echo "Watching:"
echo "  ${DEBATE_LOG}"

echo
echo "Saving outputs to:"
echo "  ${OUTPUT_DIR}"

echo
echo "Mistral     -> ${MISTRAL_OUTPUT_FILE}"
echo "Llama       -> ${LLAMA_OUTPUT_FILE}"
echo "Phi         -> ${PHI_OUTPUT_FILE}"
echo "Llama Guard -> ${GUARD_OUTPUT_FILE}"
echo "GPT-4o-mini Feedback -> ${FEEDBACK_OUTPUT_FILE}"

echo
echo "Monitor PID:"
echo "  $$"

echo
echo "============================================================"
echo


# ============================================================
# 等待本次运行 debate.log
# ============================================================

while [ ! -f "${DEBATE_LOG}" ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for debate log..."
    sleep 1
done

echo
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monitoring debate log:"
echo "${DEBATE_LOG}"
echo "============================================================"
echo


# ============================================================
# Monitor
# ============================================================

tail -n 0 -F "${DEBATE_LOG}" 2>/dev/null | \
"${PYTHON}" -u -c '
import os
import re
import sys
from datetime import datetime
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

output_dir = Path(
    os.environ["OUTPUT_DIR"]
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

files = {
    "Agent-0":
        output_dir
        / os.environ["MISTRAL_OUTPUT_FILE"],

    "Agent-1":
        output_dir
        / os.environ["LLAMA_OUTPUT_FILE"],

    "Agent-2":
        output_dir
        / os.environ["PHI_OUTPUT_FILE"],

    "Agent-Eval":
        output_dir
        / os.environ["GUARD_OUTPUT_FILE"],

    "Agent-Feedback":
        output_dir
        / os.environ["FEEDBACK_OUTPUT_FILE"],
}

names = {
    "Agent-0":
        os.environ["MISTRAL_DISPLAY_NAME"],

    "Agent-1":
        os.environ["LLAMA_DISPLAY_NAME"],

    "Agent-2":
        os.environ["PHI_DISPLAY_NAME"],

    "Agent-Eval":
        os.environ["GUARD_DISPLAY_NAME"],

    "Agent-Feedback":
        "GPT-4o-mini Feedback",
}


# ============================================================
# Regular expressions
# ============================================================

question_re = re.compile(
    r"Running debate on question:\s*(.*)"
)

response_re = re.compile(
    r"(Agent-0|Agent-1|Agent-2|Agent-Feedback)"
    r"\s+response:\s*(.*)"
)

eval_re = re.compile(
    r"(Agent-Eval)"
    r"\s+response should be blocked\?\s*(.*)"
)

timestamp_re = re.compile(
    r"^\d{4}-\d{2}-\d{2} "
    r"\d{2}:\d{2}:\d{2}"
)


# ============================================================
# Current response
# ============================================================

current_agent = None
current_question = "UNKNOWN"
buffer = []


# ============================================================
# Save response
# ============================================================

def save_response():
    global current_agent
    global buffer

    if current_agent is None:
        return

    path = files[current_agent]

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:
        f.write("\n")
        f.write("=" * 100 + "\n")

        f.write(
            "TIME: "
            + datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            + "\n"
        )

        f.write(
            f"MODEL: {names[current_agent]}\n"
        )

        f.write(
            f"QUESTION: {current_question}\n"
        )

        f.write("-" * 100 + "\n")

        for item in buffer:
            f.write(item + "\n")

        f.write("=" * 100 + "\n")
        f.flush()

    print(
        f"[SAVED] "
        f"{names[current_agent]} "
        f"-> {path}",
        flush=True,
    )

    current_agent = None
    buffer = []


# ============================================================
# Read debate log
# ============================================================

for raw_line in sys.stdin:
    line = raw_line.rstrip("\n")

    q = question_re.search(
        line
    )

    if q:
        save_response()

        current_question = q.group(1)

        continue

    m = response_re.search(
        line
    )

    if m:
        save_response()

        current_agent = m.group(1)

        first_line = m.group(2)

        buffer = []

        if first_line:
            buffer.append(
                first_line
            )

        continue

    m = eval_re.search(
        line
    )

    if m:
        save_response()

        current_agent = m.group(1)

        first_line = m.group(2)

        buffer = []

        if first_line:
            buffer.append(
                first_line
            )

        continue

    if timestamp_re.match(
        line
    ):
        save_response()

        continue

    if current_agent is not None:
        buffer.append(
            line
        )

save_response()
'