import json
import logging
import os
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_COMMON_CONFIG = (
    PROJECT_ROOT / "config" / "debate_common.yaml"
)


# ============================================================
# Logger
# ============================================================

def setup_logger(log_file_path):
    """
    Sets up a logger that writes messages to both stdout and a log file.
    """
    logger = logging.getLogger(__name__)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        file_handler = logging.FileHandler(log_file_path)

        console_handler.setLevel(logging.DEBUG)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


# ============================================================
# YAML configuration
# ============================================================

def load_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Load one YAML configuration file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(
            f"YAML root must be a mapping: {config_path}"
        )

    return config


def deep_merge(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries recursively.
    """
    result = deepcopy(base_config)

    for key, value in override_config.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(
                result[key],
                value,
            )
        else:
            result[key] = deepcopy(value)

    return result


def expand_config_value(value: Any) -> Any:
    """
    Expand environment variables and user paths.
    """
    if isinstance(value, str):
        value = os.path.expandvars(value)
        value = os.path.expanduser(value)
        return value

    if isinstance(value, dict):
        return {
            key: expand_config_value(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            expand_config_value(item)
            for item in value
        ]

    return value


def load_config(
    common_config: Optional[str] = None,
    algorithm_config: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load common configuration and optionally merge one algorithm configuration.
    """
    common_path = Path(
        common_config
        or DEFAULT_COMMON_CONFIG
    )

    config = load_yaml(
        common_path
    )

    if algorithm_config is not None:
        algorithm = load_yaml(
            Path(algorithm_config)
        )

        config = deep_merge(
            config,
            algorithm,
        )

    return expand_config_value(
        config
    )


# ============================================================
# Configuration helpers
# ============================================================

def get_config_value(
    config: Dict[str, Any],
    key_path: str,
) -> Any:
    """
    Read a nested configuration value.

    Example:
        get_config_value(
            config,
            "models.agent3.max_model_len"
        )
    """
    value = config

    for key in key_path.split("."):
        if not isinstance(value, dict):
            raise KeyError(
                f"Invalid config path: {key_path}"
            )

        if key not in value:
            raise KeyError(
                f"Config key not found: {key_path}"
            )

        value = value[key]

    return value


def require_config_value(
    config: Dict[str, Any],
    key_path: str,
) -> Any:
    """
    Read one required configuration value.
    """
    try:
        return get_config_value(
            config,
            key_path,
        )
    except KeyError as e:
        raise RuntimeError(
            f"Required config missing: {key_path}"
        ) from e


# ============================================================
# Model helpers
# ============================================================

def get_model_config(
    config: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """
    Get configuration for one local model.
    """
    model = require_config_value(
        config,
        f"models.{model_name}",
    )

    if not isinstance(model, dict):
        raise ValueError(
            f"Invalid model config: {model_name}"
        )

    return model


def get_model_url(
    config: Dict[str, Any],
    model_name: str,
) -> str:
    """
    Get the OpenAI-compatible URL of one local model.
    """
    host = str(
        require_config_value(
            config,
            "router.host",
        )
    )

    model = get_model_config(
        config,
        model_name,
    )

    return (
        f"http://{host}:{model['port']}/v1"
    )


def build_local_model_routes(
    config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Build local model routing table for the OpenAI router.
    """
    routes = {}

    for model_name in (
        "agent1",
        "agent2",
        "agent3",
        "evaluator",
    ):
        model = get_model_config(
            config,
            model_name,
        )

        routes[
            model["served_name"]
        ] = get_model_url(
            config,
            model_name,
        )

    return routes


# ============================================================
# Algorithm model helpers
# ============================================================

def build_model_spec(
    config: Dict[str, Any],
    model_name: str,
    memory_enabled: bool,
) -> str:
    """
    Build model specification used by CaRDebate.

    Example:
        openai:<served_name>:true
    """
    model = get_model_config(
        config,
        model_name,
    )

    memory = (
        "true"
        if memory_enabled
        else "false"
    )

    return (
        f"openai:{model['served_name']}:{memory}"
    )


def get_debating_models(
    config: Dict[str, Any],
):
    """
    Build debating model list from the algorithm configuration.
    """
    models = require_config_value(
        config,
        "debating_models",
    )

    if not isinstance(models, list):
        raise ValueError(
            "debating_models must be a list"
        )

    return [
        build_model_spec(
            config,
            model_name,
            True,
        )
        for model_name in models
    ]


def get_evaluator_model(
    config: Dict[str, Any],
) -> str:
    """
    Build evaluator model specification.
    """
    evaluator = require_config_value(
        config,
        "evaluator",
    )

    return build_model_spec(
        config,
        evaluator,
        False,
    )


def get_feedback_model(
    config: Dict[str, Any],
) -> str:
    """
    Build feedback generator model specification.
    """
    feedback_name = require_config_value(
        config,
        "feedback_generator",
    )

    feedback = require_config_value(
        config,
        f"remote_models.{feedback_name}",
    )

    return (
        f"openai:{feedback['served_name']}:true"
    )


# ============================================================
# Common shell configuration
# ============================================================

def build_common_shell_config(
    config: Dict[str, Any],
) -> Dict[str, str]:
    """
    Convert common YAML configuration into Bash environment variables.
    """
    project = config["project"]
    environment = config["environment"]
    model_server = config["model_server"]
    dataset = config["dataset"]
    router = config["router"]
    models = config["models"]
    output_files = config.get(
        "model_output_files",
        {}
    )
    generation = config.get(
        "generation",
        {}
    )

    if not isinstance(output_files, dict):
        raise ValueError(
            "model_output_files must be a mapping"
        )
    if not isinstance(generation, dict):
        raise ValueError(
            "generation must be a mapping"
        )

    answer_max_words = int(
        require_config_value(config, "generation.answer_max_words")
    )
    format_repair_max_words = int(
        require_config_value(config, "generation.format_repair_max_words")
    )
    if answer_max_words <= 0:
        raise ValueError(
            "generation.answer_max_words must be > 0"
        )
    if format_repair_max_words <= 0:
        raise ValueError(
            "generation.format_repair_max_words must be > 0"
        )

    agent1 = models["agent1"]
    agent2 = models["agent2"]
    agent3 = models["agent3"]
    evaluator = models["evaluator"]

    host = str(
        router["host"]
    )

    router_url = (
        f"http://{host}:{router['port']}"
    )

    local_model_routes = build_local_model_routes(
        config
    )

    values = {
        "PROJECT_DIR":
            project["project_dir"],

        "DATA_ROOT":
            project["data_root"],

        "CONDA_ENV":
            environment["conda_env"],

        "PYTHON_BIN":
            environment["conda_env"]
            + "/bin/python",

        "VLLM_ENV":
            environment["vllm_env"],

        "VLLM_BIN":
            environment["vllm_env"]
            + "/bin/vllm",

        "VLLM_PYTHON":
            environment["vllm_env"]
            + "/bin/python",

        "CUDA_HOME":
            environment["cuda_home"],

        "MODEL_SERVER_ROOT":
            model_server["model_server_root"],

        "MODEL_SERVER_LOG":
            model_server["model_server_log"],

        "MODEL_SERVER_PID":
            model_server["model_server_pid"],

        "DATASET_NAME":
            dataset["name"],

        "DATASET_PATH":
            dataset["path"],

        "ANSWER_MAX_WORDS":
            answer_max_words,

        "FORMAT_REPAIR_MAX_WORDS":
            format_repair_max_words,

        "HOST":
            host,

        "LOCAL_API_KEY":
            router["api_key"],

        "ROUTER_PORT":
            router["port"],

        "ROUTER_URL":
            router_url,

        "REMOTE_OPENAI_BASE_URL":
            router["remote_base_url"],

        "OPENAI_API_KEY":
            router["api_key"],

        "OPENAI_BASE_URL":
            router_url
            + "/v1",

        "LOCAL_MODEL_ROUTES_JSON":
            json.dumps(
                local_model_routes,
                separators=(",", ":"),
            ),

        "MODEL_OUTPUT_FILES_JSON":
            json.dumps(
                output_files,
                ensure_ascii=False,
                separators=(",", ":"),
            ),

        "AGENT1_DISPLAY_NAME":
            agent1["display_name"],

        "AGENT1_MODEL_PATH":
            agent1["model_path"],

        "AGENT1_SERVED_NAME":
            agent1["served_name"],

        "GPU_AGENT1":
            agent1["gpu"],

        "PORT_AGENT1":
            agent1["port"],

        "MEM_AGENT1":
            agent1["gpu_memory_utilization"],

        "MAX_LEN_AGENT1":
            agent1["max_model_len"],

        "AGENT1_URL":
            get_model_url(
                config,
                "agent1",
            ),

        "AGENT1_OUTPUT_FILE":
            output_files.get(
                "agent1",
                "agent1.jsonl",
            ),

        "AGENT2_DISPLAY_NAME":
            agent2["display_name"],

        "AGENT2_MODEL_PATH":
            agent2["model_path"],

        "AGENT2_SERVED_NAME":
            agent2["served_name"],

        "GPU_AGENT2":
            agent2["gpu"],

        "PORT_AGENT2":
            agent2["port"],

        "MEM_AGENT2":
            agent2["gpu_memory_utilization"],

        "MAX_LEN_AGENT2":
            agent2["max_model_len"],

        "AGENT2_URL":
            get_model_url(
                config,
                "agent2",
            ),

        "AGENT2_OUTPUT_FILE":
            output_files.get(
                "agent2",
                "agent2.jsonl",
            ),

        "AGENT3_DISPLAY_NAME":
            agent3["display_name"],

        "AGENT3_MODEL_PATH":
            agent3["model_path"],

        "AGENT3_SERVED_NAME":
            agent3["served_name"],

        "GPU_AGENT3":
            agent3["gpu"],

        "PORT_AGENT3":
            agent3["port"],

        "MEM_AGENT3":
            agent3["gpu_memory_utilization"],

        "MAX_LEN_AGENT3":
            agent3["max_model_len"],

        "AGENT3_URL":
            get_model_url(
                config,
                "agent3",
            ),

        "AGENT3_OUTPUT_FILE":
            output_files.get(
                "agent3",
                "agent3.jsonl",
            ),

        "EVALUATOR_DISPLAY_NAME":
            evaluator["display_name"],

        "EVALUATOR_MODEL_PATH":
            evaluator["model_path"],

        "EVALUATOR_SERVED_NAME":
            evaluator["served_name"],

        "GPU_EVALUATOR":
            evaluator["gpu"],

        "PORT_EVALUATOR":
            evaluator["port"],

        "MEM_EVALUATOR":
            evaluator["gpu_memory_utilization"],

        "MAX_LEN_EVALUATOR":
            evaluator["max_model_len"],

        "EVALUATOR_URL":
            get_model_url(
                config,
                "evaluator",
            ),

        "EVALUATOR_OUTPUT_FILE":
            output_files.get(
                "evaluator",
                "evaluator.jsonl",
            ),

        "FEEDBACK_OUTPUT_FILE":
            output_files.get(
                "feedback",
                "feedback.jsonl",
            ),
    }

    return {
        key: str(value)
        for key, value in values.items()
    }


# ============================================================
# Algorithm shell configuration
# ============================================================

def _require_mapping(config: Dict[str, Any], key_path: str) -> Dict[str, Any]:
    value = require_config_value(config, key_path)
    if not isinstance(value, dict):
        raise ValueError(f"{key_path} must be a mapping")
    return value


def _require_list(config: Dict[str, Any], key_path: str) -> list:
    value = require_config_value(config, key_path)
    if not isinstance(value, list):
        raise ValueError(f"{key_path} must be a list")
    return value


def _validate_probability(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _validate_positive_int(name: str, value: Any) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_nonnegative_int(name: str, value: Any) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _validate_weight_pair(
    first_name: str,
    first_value: Any,
    second_name: str,
    second_value: Any,
) -> tuple[float, float]:
    first = _validate_probability(first_name, float(first_value))
    second = _validate_probability(second_name, float(second_value))
    if abs((first + second) - 1.0) > 1e-9:
        raise ValueError(
            f"{first_name} + {second_name} must equal 1.0; "
            f"got {first + second}"
        )
    return first, second



def _validate_positive_float(name: str, value: Any) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _require_exact_protocol_value(
    config: Dict[str, Any],
    key_path: str,
    expected: Any,
) -> Any:
    """Require one YAML declaration to match the fixed CaRDebate-v7 contract.

    Fixed protocol/schema declarations remain implemented in Python.  They are
    nevertheless declared in YAML for auditability, so a changed YAML value must
    fail fast instead of appearing configurable while being silently ignored.
    """
    actual = require_config_value(config, key_path)
    if isinstance(expected, bool):
        matches = isinstance(actual, bool) and actual is expected
    else:
        matches = actual == expected
    if not matches:
        raise ValueError(
            f"{key_path} is fixed by CaRDebate v7 and must be {expected!r}; "
            f"got {actual!r}"
        )
    return actual


# Algorithm-name ablation switch. All other generation/protocol parameters stay fixed.
CARDEBATE_ABLATION_MODES = {
    # history_enabled, successful_history_only, monotonic_commit_enabled
    "cardebate": (True, True, True),
    "cardebate_all_history": (True, False, True),
    "cardebate_no_history": (False, True, True),
    "cardebate_without_rollback": (True, True, False),
}


def build_cardebate_run_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the runtime arguments for CaRDebate v7 from a compact YAML config.

    YAML remains the source of experiment parameters.  Fixed protocol invariants
    and backend-specific generation details are implemented here rather than
    repeated as pseudo-hyperparameters in YAML.

    This function intentionally preserves the runtime structures expected by
    ``run.py`` and the existing shell scripts, so the simplification is confined
    to the YAML -> runtime translation boundary.
    """
    algo_name = str(require_config_value(config, "algo_name")).strip().lower()
    if algo_name not in CARDEBATE_ABLATION_MODES:
        raise ValueError(
            f"Unsupported CaRDebate algo_name={algo_name!r}; choose one of "
            f"{list(CARDEBATE_ABLATION_MODES)}"
        )

    (
        history_enabled,
        successful_history_only,
        monotonic_commit_enabled,
    ) = CARDEBATE_ABLATION_MODES[algo_name]

    mechanism = _require_mapping(config, "mechanism")
    expected_version = (
        "evaluator_safe_peer_bayesian_question_v7_stm_success_category_memory"
    )
    version = str(require_config_value(config, "mechanism.version"))
    if version != expected_version:
        raise ValueError(
            f"Unsupported mechanism.version: {version!r}; expected {expected_version!r}"
        )

    if not bool(require_config_value(config, "mechanism.enabled")):
        raise ValueError("mechanism.enabled must be true")
    if not bool(require_config_value(config, "mechanism.require_three_rounds")):
        raise ValueError("mechanism.require_three_rounds must be true")

    rounds = int(require_config_value(config, "debate_rounds"))
    if rounds != 3:
        raise ValueError("CaRDebate v7 requires debate_rounds: 3")

    # Keep the short round-policy declaration in YAML because it is useful audit
    # information, but do not require dozens of lower-level fixed protocol flags.
    round_policy = _require_mapping(config, "mechanism.round_policy")
    expected_round_policy = {
        "round1": "independent_answer_evaluate_critique",
        "round2": "revision_evaluate_critique",
        "round3": "final_answer_evaluate_feedback",
    }
    for key, expected in expected_round_policy.items():
        actual = str(round_policy[key])
        if actual != expected:
            raise ValueError(
                f"mechanism.round_policy.{key} must be {expected!r}; got {actual!r}"
            )

    # Tunable / experiment-relevant sections.
    experiment = _require_mapping(config, "experiment")
    reliability = _require_mapping(config, "mechanism.reliability")
    routing = _require_mapping(config, "mechanism.routing")
    history = _require_mapping(config, "mechanism.history_assistance")
    revision = _require_mapping(config, "mechanism.revision")
    memory = _require_mapping(config, "mechanism.memory")
    noisy = _require_mapping(config, "mechanism.evaluator.noisy_observation")
    feedback = _require_mapping(config, "mechanism.feedback")
    monotonic = _require_mapping(config, "mechanism.monotonic_commit")
    format_retry = _require_mapping(config, "mechanism.generation.format_retry")
    root_generation = _require_mapping(config, "generation")
    debater_generation = _require_mapping(config, "generation.debater")
    evaluator_generation = _require_mapping(config, "generation.evaluator")
    feedback_generation = _require_mapping(config, "generation.feedback")
    adaptation = _require_mapping(config, "adaptation")

    # ------------------------------------------------------------------
    # Memory / history
    # ------------------------------------------------------------------
    memory_top_k = _validate_positive_int(
        "mechanism.memory.memory_top_k", memory["memory_top_k"]
    )
    memory_episode_top_k = _validate_positive_int(
        "mechanism.memory.episode_top_k", memory["episode_top_k"]
    )
    memory_max_rules_per_episode = _validate_positive_int(
        "mechanism.memory.max_rules_per_episode", memory["max_rules_per_episode"]
    )
    memory_max_rules_stored_per_episode = _validate_positive_int(
        "mechanism.memory.max_rules_stored_per_episode",
        memory["max_rules_stored_per_episode"],
    )

    if memory_top_k > 5:
        raise ValueError("mechanism.memory.memory_top_k must be <= 5")
    if memory_episode_top_k > 20:
        raise ValueError("mechanism.memory.episode_top_k must be <= 20")
    if memory_max_rules_per_episode > 5:
        raise ValueError("mechanism.memory.max_rules_per_episode must be <= 5")
    if memory_max_rules_stored_per_episode > 20:
        raise ValueError(
            "mechanism.memory.max_rules_stored_per_episode must be <= 20"
        )

    history_ranges = {
        "safe_peer": (
            int(history["safe_peer_history_min_items"]),
            int(history["safe_peer_history_max_items"]),
        ),
        "no_safe_peer": (
            int(history["no_safe_peer_history_min_items"]),
            int(history["no_safe_peer_history_max_items"]),
        ),
        "cross_category": (
            int(history["cross_category_history_min_items"]),
            int(history["cross_category_history_max_items"]),
        ),
    }
    for branch, (min_items, max_items) in history_ranges.items():
        if min_items < 1 or max_items < min_items:
            raise ValueError(
                f"{branch} history range must satisfy 1 <= min_items <= max_items"
            )
        if max_items > memory_top_k:
            raise ValueError(
                f"{branch} history max_items ({max_items}) must not exceed "
                f"memory_top_k ({memory_top_k})"
            )

    memory_min_relevance = _validate_probability(
        "mechanism.memory.min_question_similarity",
        memory["min_question_similarity"],
    )
    memory_min_support = _validate_probability(
        "mechanism.memory.min_feedback_support",
        memory["min_feedback_support"],
    )
    memory_dedup_threshold = _validate_probability(
        "mechanism.memory.semantic_dedup_threshold",
        memory["semantic_dedup_threshold"],
    )
    memory_min_marginal_utility = _validate_probability(
        "mechanism.memory.min_marginal_utility",
        memory["min_marginal_utility"],
    )
    memory_candidate_oversample_factor = _validate_positive_int(
        "mechanism.memory.candidate_oversample_factor",
        memory["candidate_oversample_factor"],
    )

    memory_ranking = _require_mapping(config, "mechanism.memory.ranking_weights")
    memory_relevance_weight = _validate_probability(
        "mechanism.memory.ranking_weights.relevance",
        memory_ranking["relevance"],
    )
    memory_support_weight = _validate_probability(
        "mechanism.memory.ranking_weights.support",
        memory_ranking["support"],
    )
    memory_outcome_priority_weight = _validate_probability(
        "mechanism.memory.ranking_weights.outcome_priority",
        memory_ranking["outcome_priority"],
    )
    if abs(
        memory_relevance_weight
        + memory_support_weight
        + memory_outcome_priority_weight
        - 1.0
    ) > 1e-9:
        raise ValueError(
            "mechanism.memory.ranking_weights must sum to 1.0"
        )

    # ------------------------------------------------------------------
    # Reliability / evaluator observation
    # ------------------------------------------------------------------
    reliability_prior_alpha = float(reliability["prior_alpha"])
    reliability_prior_beta = float(reliability["prior_beta"])
    if reliability_prior_alpha <= 0.0 or reliability_prior_beta <= 0.0:
        raise ValueError("mechanism.reliability prior_alpha/prior_beta must be > 0")

    reliability_credible_level = float(reliability["credible_level"])
    if not 0.5 <= reliability_credible_level < 1.0:
        raise ValueError(
            "mechanism.reliability.credible_level must be in [0.5, 1.0)"
        )

    reliability_similar_history_k = _validate_nonnegative_int(
        "mechanism.reliability.similar_history_k",
        reliability["similar_history_k"],
    )
    reliability_min_similarity = _validate_probability(
        "mechanism.reliability.min_similarity",
        reliability["min_similarity"],
    )

    unsafe_sensitivity = _validate_probability(
        "mechanism.evaluator.noisy_observation.unsafe_sensitivity",
        noisy["unsafe_sensitivity"],
    )
    safe_specificity = _validate_probability(
        "mechanism.evaluator.noisy_observation.safe_specificity",
        noisy["safe_specificity"],
    )
    evaluator_prior_floor = _validate_probability(
        "mechanism.evaluator.noisy_observation.prior_floor",
        noisy["prior_floor"],
    )
    evaluator_prior_ceiling = _validate_probability(
        "mechanism.evaluator.noisy_observation.prior_ceiling",
        noisy["prior_ceiling"],
    )
    if evaluator_prior_floor > evaluator_prior_ceiling:
        raise ValueError("evaluator prior_floor must not exceed prior_ceiling")

    max_safe_peers = _validate_nonnegative_int(
        "mechanism.routing.max_safe_peers", routing["max_safe_peers"]
    )

    # ------------------------------------------------------------------
    # Feedback / monotonic commit
    # ------------------------------------------------------------------
    feedback_min_quality = _validate_probability(
        "mechanism.feedback.min_quality", feedback["min_quality"]
    )
    feedback_min_confidence = _validate_probability(
        "mechanism.feedback.min_confidence", feedback["min_confidence"]
    )

    feedback_support_weights = _require_mapping(
        config, "mechanism.feedback.support_score_weights"
    )
    feedback_support_quality_weight, feedback_support_confidence_weight = (
        _validate_weight_pair(
            "mechanism.feedback.support_score_weights.quality",
            feedback_support_weights["quality"],
            "mechanism.feedback.support_score_weights.confidence",
            feedback_support_weights["confidence"],
        )
    )

    monotonic_safety_score_tolerance = _validate_probability(
        "mechanism.monotonic_commit.safety_score_tolerance",
        monotonic["safety_score_tolerance"],
    )
    monotonic_safety_coverage_tolerance = _validate_probability(
        "mechanism.monotonic_commit.safety_coverage_tolerance",
        monotonic["safety_coverage_tolerance"],
    )

    # ------------------------------------------------------------------
    # Experiment size / retry policy
    # ------------------------------------------------------------------
    raw_max_total_debates = experiment["max_total_debates"]
    if raw_max_total_debates is None:
        max_total_debates = None
    else:
        max_total_debates = _validate_positive_int(
            "experiment.max_total_debates", raw_max_total_debates
        )

    format_retry_max_retries = _validate_nonnegative_int(
        "mechanism.generation.format_retry.max_retries",
        format_retry["max_retries"],
    )
    format_retry_reasons = _require_list(
        config, "mechanism.generation.format_retry.retry_on"
    )
    format_retry_reasons = [
        str(item).strip() for item in format_retry_reasons if str(item).strip()
    ]
    if not format_retry_reasons:
        raise ValueError(
            "mechanism.generation.format_retry.retry_on must be a non-empty list"
        )

    # ------------------------------------------------------------------
    # Generation
    #
    # Current baseline is deterministic.  Backend-specific API fields are
    # synthesized here so run.py does not need to change.
    # ------------------------------------------------------------------
    deterministic = bool(root_generation["deterministic"])

    answer_max_words = _validate_positive_int(
        "generation.answer_max_words", root_generation["answer_max_words"]
    )
    format_repair_max_words = _validate_positive_int(
        "generation.format_repair_max_words",
        root_generation["format_repair_max_words"],
    )
    llm_request_max_retries = _validate_nonnegative_int(
        "generation.llm_request_max_retries",
        root_generation["llm_request_max_retries"],
    )

    def resolve_generation_role(
        role_name: str,
        role: Dict[str, Any],
    ) -> Dict[str, Any]:
        temperature = float(role["temperature"])
        top_p = float(role["top_p"])
        do_sample = bool(role["do_sample"])
        if temperature < 0.0:
            raise ValueError(f"generation.{role_name}.temperature must be >= 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"generation.{role_name}.top_p must be in (0, 1]")
        if deterministic and (temperature != 0.0 or top_p != 1.0 or do_sample):
            raise ValueError(
                f"generation.{role_name} conflicts with generation.deterministic: true"
            )
        max_tokens = _validate_positive_int(
            f"generation.{role_name}.max_tokens",
            role["max_tokens"],
        )
        return {
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            # HF-style wrappers consume max_new_tokens; OpenAI/vLLM wrappers
            # consume max_tokens.  One YAML token budget feeds both interfaces.
            "max_new_tokens": max_tokens,
            "max_tokens": max_tokens,
            "truncation": True,
            "return_full_text": False,
        }

    debater_generation_runtime = resolve_generation_role(
        "debater", debater_generation
    )
    evaluator_generation_runtime = resolve_generation_role(
        "evaluator", evaluator_generation
    )
    feedback_generation_runtime = resolve_generation_role(
        "feedback", feedback_generation
    )

    remote_guard = _require_mapping(config, "generation.remote_llamaguard")
    remote_guard_temperature = float(remote_guard["temperature"])
    remote_guard_top_p = float(remote_guard["top_p"])
    if remote_guard_temperature < 0.0:
        raise ValueError("generation.remote_llamaguard.temperature must be >= 0")
    if not 0.0 < remote_guard_top_p <= 1.0:
        raise ValueError("generation.remote_llamaguard.top_p must be in (0, 1]")
    remote_guard_config = {
        "temperature": remote_guard_temperature,
        "top_p": remote_guard_top_p,
        "max_tokens": _validate_positive_int(
            "generation.remote_llamaguard.max_tokens",
            remote_guard["max_tokens"],
        ),
    }
    local_guard = _require_mapping(config, "generation.local_llamaguard")
    local_guard_config = {
        "max_new_tokens": _validate_positive_int(
            "generation.local_llamaguard.max_new_tokens",
            local_guard["max_new_tokens"],
        ),
        "pad_token_id": _validate_nonnegative_int(
            "generation.local_llamaguard.pad_token_id",
            local_guard["pad_token_id"],
        ),
    }

    # LlamaGuard backend details are implementation constants, not experiment
    # hyperparameters.  They remain in the runtime structure only because the
    # existing run.py supports dedicated local/remote guard wrappers.
    generation_config = {
        "deterministic": deterministic,
        "answer_max_words": answer_max_words,
        "format_repair_max_words": format_repair_max_words,
        "llm_request_max_retries": llm_request_max_retries,
        "debater": debater_generation_runtime,
        "evaluator": evaluator_generation_runtime,
        "feedback": feedback_generation_runtime,
        "remote_llamaguard": remote_guard_config,
        "local_llamaguard": local_guard_config,
    }

    # ------------------------------------------------------------------
    # Optional PEFT
    #
    # For the normal baseline (peft_memory: false), dormant training settings
    # do not need to clutter YAML.  If PEFT is enabled, its parameters become
    # experiment parameters and are required explicitly.
    # ------------------------------------------------------------------
    peft_enabled = bool(adaptation["peft_memory"])

    if peft_enabled:
        peft_lora = _require_mapping(config, "adaptation.lora")
        peft_training = _require_mapping(config, "adaptation.training")

        peft_dropout = _validate_probability(
            "adaptation.lora.dropout", peft_lora["dropout"]
        )
        if peft_dropout >= 1.0:
            raise ValueError("adaptation.lora.dropout must be in [0, 1)")

        peft_target_modules = str(peft_lora["target_modules"]).strip()
        if not peft_target_modules:
            raise ValueError("adaptation.lora.target_modules must be non-empty")

        peft_bias = str(peft_lora["bias"]).strip()
        if peft_bias not in {"none", "all", "lora_only"}:
            raise ValueError(
                "adaptation.lora.bias must be one of: none, all, lora_only"
            )

        peft_save_strategy = str(peft_training["save_strategy"]).strip()
        peft_report_to = str(peft_training["report_to"]).strip()
        if not peft_save_strategy or not peft_report_to:
            raise ValueError(
                "adaptation.training.save_strategy/report_to must be non-empty"
            )

        peft_config = {
            "enabled": True,
            "train_steps": _validate_positive_int(
                "adaptation.train_steps", adaptation["train_steps"]
            ),
            "directory": str(adaptation["peft_directory"]),
            "lora": {
                "r": _validate_positive_int(
                    "adaptation.lora.r", peft_lora["r"]
                ),
                "alpha": _validate_positive_int(
                    "adaptation.lora.alpha", peft_lora["alpha"]
                ),
                "dropout": peft_dropout,
                "target_modules": peft_target_modules,
                "bias": peft_bias,
            },
            "training": {
                "num_train_epochs": _validate_positive_float(
                    "adaptation.training.num_train_epochs",
                    peft_training["num_train_epochs"],
                ),
                "per_device_train_batch_size": _validate_positive_int(
                    "adaptation.training.per_device_train_batch_size",
                    peft_training["per_device_train_batch_size"],
                ),
                "learning_rate": _validate_positive_float(
                    "adaptation.training.learning_rate",
                    peft_training["learning_rate"],
                ),
                "max_length": _validate_positive_int(
                    "adaptation.training.max_length",
                    peft_training["max_length"],
                ),
                "logging_steps": _validate_positive_int(
                    "adaptation.training.logging_steps",
                    peft_training["logging_steps"],
                ),
                "save_strategy": peft_save_strategy,
                "report_to": peft_report_to,
            },
        }
    else:
        # Dormant values keep the existing runtime/shell interface stable.
        peft_config = {
            "enabled": False,
            "train_steps": 10,
            "directory": "",
            "lora": {
                "r": 8,
                "alpha": 32,
                "dropout": 0.10,
                "target_modules": "all-linear",
                "bias": "none",
            },
            "training": {
                "num_train_epochs": 5.0,
                "per_device_train_batch_size": 1,
                "learning_rate": 0.00005,
                "max_length": 512,
                "logging_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
            },
        }

    # Runtime debate configuration expected by debate.py/run.py.
    # Fixed CaRDebate-v7 semantics are encoded as constants here rather than
    # repeated as dozens of YAML booleans.
    debate_config = {
        "memory_top_k": memory_top_k,
        "memory_episode_top_k": memory_episode_top_k,
        "memory_max_rules_per_episode": memory_max_rules_per_episode,
        "memory_max_rules_stored_per_episode": memory_max_rules_stored_per_episode,
        "memory_min_relevance": memory_min_relevance,
        "memory_min_support": memory_min_support,
        "memory_semantic_dedup_enabled": bool(memory["semantic_dedup_enabled"]),
        "memory_dedup_threshold": memory_dedup_threshold,
        "memory_min_marginal_utility": memory_min_marginal_utility,
        "memory_candidate_oversample_factor": memory_candidate_oversample_factor,
        "memory_relevance_weight": memory_relevance_weight,
        "memory_support_weight": memory_support_weight,
        "memory_outcome_priority_weight": memory_outcome_priority_weight,
        "safe_peer_history_min_items": history_ranges["safe_peer"][0],
        "safe_peer_history_max_items": history_ranges["safe_peer"][1],
        "no_safe_peer_history_min_items": history_ranges["no_safe_peer"][0],
        "no_safe_peer_history_max_items": history_ranges["no_safe_peer"][1],
        "cross_category_history_min_items": history_ranges["cross_category"][0],
        "cross_category_history_max_items": history_ranges["cross_category"][1],
        "successful_history_only": successful_history_only,
        "history_enabled": history_enabled,
        "feedback_enabled": history_enabled,
        "history_mode": algo_name,
        "prefer_unsafe_to_safe_history": True,
        "enable_verified_repair_memory": bool(
            revision["enable_verified_repair_memory"]
        ),
        "format_retry_max_retries": format_retry_max_retries,
        "format_retry_reasons": format_retry_reasons,
        "reliability_prior_alpha": reliability_prior_alpha,
        "reliability_prior_beta": reliability_prior_beta,
        "reliability_credible_level": reliability_credible_level,
        "reliability_similar_history_k": reliability_similar_history_k,
        "reliability_min_similarity": reliability_min_similarity,
        "use_expected_safety_mass": bool(
            reliability["use_expected_safety_mass"]
        ),
        "unsafe_sensitivity": unsafe_sensitivity,
        "safe_specificity": safe_specificity,
        "evaluator_prior_floor": evaluator_prior_floor,
        "evaluator_prior_ceiling": evaluator_prior_ceiling,
        "max_safe_peers": max_safe_peers,
        "feedback_min_quality": feedback_min_quality,
        "feedback_min_confidence": feedback_min_confidence,
        "feedback_support_quality_weight": feedback_support_quality_weight,
        "feedback_support_confidence_weight": feedback_support_confidence_weight,
        "monotonic_commit_enabled": monotonic_commit_enabled,
        "monotonic_safety_score_tolerance": monotonic_safety_score_tolerance,
        "monotonic_safety_coverage_tolerance": monotonic_safety_coverage_tolerance,
    }

    debating_models = get_debating_models(config)
    if len(debating_models) != 3:
        raise ValueError("debating_models must contain exactly 3 models")

    dataset = _require_mapping(config, "dataset")
    dataset_name = str(dataset["name"])
    dataset_path = str(dataset["path"])
    if not dataset_name or not dataset_path:
        raise ValueError("dataset.name and dataset.path must be non-empty")

    output_files = config.get("model_output_files", {})
    if not isinstance(output_files, dict):
        raise ValueError("model_output_files must be a mapping")

    return {
        "debater_models": debating_models,
        "evaluator_model": get_evaluator_model(config),
        "datasets": [f"{dataset_name}:{dataset_path}"],
        "feedback_generator": get_feedback_model(config),
        "textual_memory_index": require_config_value(
            config, "textual_memory_index"
        ),
        "model_output_files": dict(output_files),
        "llamaguard_cuda_device": int(
            require_config_value(config, "models.evaluator.gpu")
        ),
        "generation_config": generation_config,
        "peft_config": peft_config,
        "debate_config": debate_config,
        "abort_question_on_exhaustion": bool(
            format_retry["abort_question_on_exhaustion"]
        ),
        "continue_experiment_after_question_abort": bool(
            format_retry["continue_experiment_after_question_abort"]
        ),
        "max_total_debates": max_total_debates,
    }


def build_algorithm_shell_config(
    config: Dict[str, Any],
) -> Dict[str, str]:
    """Export YAML-resolved algorithm settings for legacy shell tooling.

    The active Python runner no longer reads these variables as hyperparameter
    fallbacks.  They are retained only so existing shell launch scripts can inspect
    or export the same YAML-resolved values without creating a second source of truth.
    """
    runtime = build_cardebate_run_kwargs(config)
    debate = runtime["debate_config"]
    generation = runtime["generation_config"]
    peft = runtime["peft_config"]

    values = {
        "ALGO_NAME": require_config_value(config, "algo_name"),
        "DEBATE_MODEL_0": runtime["debater_models"][0],
        "DEBATE_MODEL_1": runtime["debater_models"][1],
        "DEBATE_MODEL_2": runtime["debater_models"][2],
        "EVALUATOR_MODEL": runtime["evaluator_model"],
        "FEEDBACK_MODEL": runtime["feedback_generator"],
        "DEBATE_ROUNDS": require_config_value(config, "debate_rounds"),
        "TEXTUAL_MEMORY_INDEX": runtime["textual_memory_index"],
        "MEMORY_TOP_K": debate["memory_top_k"],
        "MEMORY_EPISODE_TOP_K": debate["memory_episode_top_k"],
        "MEMORY_MAX_RULES_PER_EPISODE": debate["memory_max_rules_per_episode"],
        "MEMORY_MAX_RULES_STORED_PER_EPISODE": debate["memory_max_rules_stored_per_episode"],
        "MEMORY_MIN_RELEVANCE": debate["memory_min_relevance"],
        "MEMORY_MIN_SUPPORT": debate["memory_min_support"],
        "MEMORY_SEMANTIC_DEDUP_ENABLED": debate["memory_semantic_dedup_enabled"],
        "MEMORY_DEDUP_THRESHOLD": debate["memory_dedup_threshold"],
        "MEMORY_MIN_MARGINAL_UTILITY": debate["memory_min_marginal_utility"],
        "MEMORY_CANDIDATE_OVERSAMPLE_FACTOR": debate["memory_candidate_oversample_factor"],
        "MEMORY_RELEVANCE_WEIGHT": debate["memory_relevance_weight"],
        "MEMORY_SUPPORT_WEIGHT": debate["memory_support_weight"],
        "MEMORY_OUTCOME_PRIORITY_WEIGHT": debate["memory_outcome_priority_weight"],
        "SAFE_PEER_HISTORY_MIN_ITEMS": debate["safe_peer_history_min_items"],
        "SAFE_PEER_HISTORY_MAX_ITEMS": debate["safe_peer_history_max_items"],
        "NO_SAFE_PEER_HISTORY_MIN_ITEMS": debate["no_safe_peer_history_min_items"],
        "NO_SAFE_PEER_HISTORY_MAX_ITEMS": debate["no_safe_peer_history_max_items"],
        "CROSS_CATEGORY_HISTORY_MIN_ITEMS": debate["cross_category_history_min_items"],
        "CROSS_CATEGORY_HISTORY_MAX_ITEMS": debate["cross_category_history_max_items"],
        "SUCCESSFUL_HISTORY_ONLY": debate["successful_history_only"],
        "PREFER_UNSAFE_TO_SAFE_HISTORY": debate["prefer_unsafe_to_safe_history"],
        "ENABLE_VERIFIED_REPAIR_MEMORY": debate["enable_verified_repair_memory"],
        "FORMAT_RETRY_MAX_RETRIES": debate["format_retry_max_retries"],
        "FORMAT_RETRY_REASONS_JSON": json.dumps(debate["format_retry_reasons"]),
        "ABORT_QUESTION_ON_EXHAUSTION": runtime["abort_question_on_exhaustion"],
        "CONTINUE_EXPERIMENT_AFTER_QUESTION_ABORT": runtime["continue_experiment_after_question_abort"],
        "RELIABILITY_PRIOR_ALPHA": debate["reliability_prior_alpha"],
        "RELIABILITY_PRIOR_BETA": debate["reliability_prior_beta"],
        "RELIABILITY_CREDIBLE_LEVEL": debate["reliability_credible_level"],
        "RELIABILITY_SIMILAR_HISTORY_K": debate["reliability_similar_history_k"],
        "RELIABILITY_MIN_SIMILARITY": debate["reliability_min_similarity"],
        "RELIABILITY_USE_EXPECTED_SAFETY_MASS": debate["use_expected_safety_mass"],
        "MAX_SAFE_PEERS": debate["max_safe_peers"],
        "EVALUATOR_UNSAFE_SENSITIVITY": debate["unsafe_sensitivity"],
        "EVALUATOR_SAFE_SPECIFICITY": debate["safe_specificity"],
        "EVALUATOR_PRIOR_FLOOR": debate["evaluator_prior_floor"],
        "EVALUATOR_PRIOR_CEILING": debate["evaluator_prior_ceiling"],
        "FEEDBACK_MIN_QUALITY": debate["feedback_min_quality"],
        "FEEDBACK_MIN_CONFIDENCE": debate["feedback_min_confidence"],
        "FEEDBACK_SUPPORT_QUALITY_WEIGHT": debate["feedback_support_quality_weight"],
        "FEEDBACK_SUPPORT_CONFIDENCE_WEIGHT": debate["feedback_support_confidence_weight"],
        "MONOTONIC_SAFETY_SCORE_TOLERANCE": debate["monotonic_safety_score_tolerance"],
        "MONOTONIC_SAFETY_COVERAGE_TOLERANCE": debate["monotonic_safety_coverage_tolerance"],
        "DETERMINISTIC": generation["deterministic"],
        "DEBATER_MAX_TOKENS": generation["debater"]["max_tokens"],
        "DEBATER_TEMPERATURE": generation["debater"]["temperature"],
        "DEBATER_TOP_P": generation["debater"]["top_p"],
        "DEBATER_DO_SAMPLE": generation["debater"]["do_sample"],
        "EVALUATOR_MAX_TOKENS": generation["evaluator"]["max_tokens"],
        "EVALUATOR_TEMPERATURE": generation["evaluator"]["temperature"],
        "EVALUATOR_TOP_P": generation["evaluator"]["top_p"],
        "EVALUATOR_DO_SAMPLE": generation["evaluator"]["do_sample"],
        "FEEDBACK_MAX_TOKENS": generation["feedback"]["max_tokens"],
        "FEEDBACK_TEMPERATURE": generation["feedback"]["temperature"],
        "FEEDBACK_TOP_P": generation["feedback"]["top_p"],
        "FEEDBACK_DO_SAMPLE": generation["feedback"]["do_sample"],
        "ANSWER_MAX_WORDS": generation["answer_max_words"],
        "FORMAT_REPAIR_MAX_WORDS": generation["format_repair_max_words"],
        "LLM_REQUEST_MAX_RETRIES": generation["llm_request_max_retries"],
        "REMOTE_LLAMAGUARD_TEMPERATURE": generation["remote_llamaguard"]["temperature"],
        "REMOTE_LLAMAGUARD_TOP_P": generation["remote_llamaguard"]["top_p"],
        "REMOTE_LLAMAGUARD_MAX_TOKENS": generation["remote_llamaguard"]["max_tokens"],
        "LOCAL_LLAMAGUARD_MAX_NEW_TOKENS": generation["local_llamaguard"]["max_new_tokens"],
        "LOCAL_LLAMAGUARD_PAD_TOKEN_ID": generation["local_llamaguard"]["pad_token_id"],
        "PEFT_MEMORY": peft["enabled"],
        "PEFT_TRAIN_STEPS": peft["train_steps"],
        "PEFT_DIRECTORY": peft["directory"],
        "PEFT_LORA_R": peft["lora"]["r"],
        "PEFT_LORA_ALPHA": peft["lora"]["alpha"],
        "PEFT_LORA_DROPOUT": peft["lora"]["dropout"],
        "PEFT_LORA_TARGET_MODULES": peft["lora"]["target_modules"],
        "PEFT_LORA_BIAS": peft["lora"]["bias"],
        "PEFT_NUM_TRAIN_EPOCHS": peft["training"]["num_train_epochs"],
        "PEFT_TRAIN_BATCH_SIZE": peft["training"]["per_device_train_batch_size"],
        "PEFT_LEARNING_RATE": peft["training"]["learning_rate"],
        "PEFT_MAX_LENGTH": peft["training"]["max_length"],
        "PEFT_LOGGING_STEPS": peft["training"]["logging_steps"],
        "MAX_TOTAL_DEBATES": "" if runtime["max_total_debates"] is None else runtime["max_total_debates"],
        "MECHANISM_VERSION": require_config_value(config, "mechanism.version"),
    }
    return {key: str(value) for key, value in values.items()}

def build_shell_config(
    config: Dict[str, Any],
    include_algorithm: bool = False,
) -> Dict[str, str]:
    """
    Build Bash environment variables.

    Common variables are always exported.
    Algorithm variables are exported only when algorithm config is provided.
    """
    values = build_common_shell_config(
        config
    )

    if include_algorithm:
        values.update(
            build_algorithm_shell_config(
                config
            )
        )

    return values


def print_shell_config(
    common_config,
    algorithm_config=None,
):
    """
    Print Bash export statements.
    """
    config = load_config(
        common_config=common_config,
        algorithm_config=algorithm_config,
    )

    values = build_shell_config(
        config,
        include_algorithm=(
            algorithm_config is not None
        ),
    )

    for key, value in values.items():
        print(
            "export "
            + key
            + "="
            + shlex.quote(value)
        )


# ============================================================
# Command line
# ============================================================

def main():
    if (
        len(sys.argv) in (3, 4)
        and sys.argv[1] == "--shell"
    ):
        common_config = sys.argv[2]

        algorithm_config = (
            sys.argv[3]
            if len(sys.argv) == 4
            else None
        )

        print_shell_config(
            common_config,
            algorithm_config,
        )

        return

    if (
        len(sys.argv) == 3
        and sys.argv[1] == "--get"
    ):
        config = load_config()

        value = get_config_value(
            config,
            sys.argv[2],
        )

        if isinstance(
            value,
            (dict, list),
        ):
            print(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(value)

        return

    print("Usage:")

    print(
        "  python utils.py --shell "
        "<common_config.yaml>"
    )

    print(
        "  python utils.py --shell "
        "<common_config.yaml> "
        "<algorithm_config.yaml>"
    )

    print(
        "  python utils.py --get "
        "<config.path>"
    )

    sys.exit(1)


if __name__ == "__main__":
    main()
