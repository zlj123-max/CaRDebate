"""Top-level runner for the three-round CaRDebate v7 protocol.

The debate has exactly three answer rounds. Evaluation follows every answer
round; conditional critique is embedded after Round 1 and Round 2; feedback
learning follows the Round-3 final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .agents import DebateAgent, EvalAgent, FeedbackAgent, PEFTDebateAgent
from .confidence import (
    agent_stats_need_rebuild,
    default_agent_stats,
    record_r1_observations,
)
from .dataloader import load_datasets
from .debate import AnswerGenerationExhausted, Debate, PROTOCOL_VERSION
from .debate_prompts import (
    debate_agent_prompt, evaluator_prompt, feedback_prompt,
    feedback_prompt_all_history,
)
from .llm import AnyGoogleGenerativeAI, AnyHuggingFace, AnyOpenAILLM, LlamaGuard
from .memory import LongTermMemory, VectorStoreMemory
from .metrics import calculate_debate_metrics, log_results_to_swanlab
from .utils import setup_logger


load_dotenv()


def _normalise_output_mapping(value) -> dict[str, str]:
    """Return a cleaned model-output mapping from dict or JSON text."""
    if value is None:
        raw = os.environ.get("MODEL_OUTPUT_FILES_JSON")
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "MODEL_OUTPUT_FILES_JSON must contain a JSON object"
            ) from exc

    if not isinstance(value, Mapping):
        raise TypeError(
            "model_output_files must be a mapping such as "
            "{'mistral': 'mistral-7b.log'}"
        )

    result: dict[str, str] = {}
    for key, filename in value.items():
        key_text = str(key).strip()
        filename_text = str(filename).strip()
        if not key_text or not filename_text:
            raise ValueError(
                "model_output_files keys and filenames must be non-empty"
            )
        # Files live under model_output_dir. Reject path traversal / absolute paths.
        file_path = Path(filename_text)
        if file_path.is_absolute() or file_path.name != filename_text:
            raise ValueError(
                f"model output filename must be a basename, got {filename_text!r}"
            )
        result[key_text] = filename_text
    return result


def _model_aliases(model_spec: str, *, agent_name: str, role: str) -> list[str]:
    """Return stable aliases that may address one model-output file."""
    _model_type, model_name, _use_chat = _parse_model_str(model_spec)
    lower = str(model_name).lower()
    basename = Path(str(model_name).rstrip("/")).name.lower()

    aliases = [agent_name, agent_name.lower(), basename, lower]
    if role == "evaluator":
        aliases.extend(["guard", "evaluator", "eval"])
    elif role == "feedback":
        aliases.extend(["feedback", "feedback_generator"])
    else:
        if "mistral" in lower:
            aliases.append("mistral")
        if "llama" in lower:
            aliases.append("llama")
        if "phi" in lower:
            aliases.append("phi")
        if "gemma" in lower:
            aliases.append("gemma")
        if "qwen" in lower:
            aliases.append("qwen")
        if agent_name.startswith("Agent-"):
            aliases.append(agent_name.split("-", 1)[1])

    # Stable de-duplication.
    return list(dict.fromkeys(alias for alias in aliases if alias))


def _default_output_filename(model_spec: str, agent_name: str) -> str:
    """Derive a readable JSONL filename when no explicit mapping is given."""
    _model_type, model_name, _use_chat = _parse_model_str(model_spec)
    basename = Path(str(model_name).rstrip("/")).name or agent_name
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", basename).strip("-._")
    if not stem:
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", agent_name)
    return f"{stem}.jsonl"


def _resolve_model_output_files(
    configured,
    debater_models,
    evaluator_model,
    feedback_generator,
) -> dict[str, str]:
    """Resolve user-facing model aliases to internal Agent-* names.

    Explicit mappings win. Unspecified agents get readable model-derived JSONL
    filenames, so output filenames are model-centric rather than Agent-0-centric.
    """
    mapping = _normalise_output_mapping(configured)
    lower_mapping = {str(k).lower(): v for k, v in mapping.items()}
    resolved: dict[str, str] = {}

    def choose(model_spec: str, agent_name: str, role: str) -> None:
        filename = None
        for alias in _model_aliases(
            model_spec, agent_name=agent_name, role=role
        ):
            if alias in mapping:
                filename = mapping[alias]
                break
            lowered = alias.lower()
            if lowered in lower_mapping:
                filename = lower_mapping[lowered]
                break
        if filename is None:
            filename = _default_output_filename(model_spec, agent_name)
        resolved[agent_name] = filename

    for index, model_spec in enumerate(debater_models):
        choose(str(model_spec), f"Agent-{index}", "debater")

    choose(str(evaluator_model), "Agent-Eval", "evaluator")
    if feedback_generator is not None:
        choose(str(feedback_generator), "Agent-Feedback", "feedback")

    # Avoid accidental interleaving of two agents in one file unless the caller
    # explicitly chose the same filename for those agents. Explicit sharing is
    # allowed but logged by the caller-facing mapping itself.
    return resolved


def _is_llamaguard_model(model_name: str) -> bool:
    return "llama-guard" in str(model_name).lower()


def _extract_llamaguard_verdict(raw_output) -> Optional[str]:
    if isinstance(raw_output, tuple) and len(raw_output) >= 2:
        raw_output = raw_output[1]
    text = str(raw_output or "").strip().lower()
    if not text:
        return None
    first = text.splitlines()[0].strip(" \t-*`_\"'[](){}:")
    if first.startswith("unsafe"):
        return "unsafe"
    if first.startswith("safe"):
        return "safe"
    return None


def _extract_llamaguard_reason_codes(raw_output) -> list[str]:
    codes = re.findall(r"\bS(?:1[0-4]|[1-9])\b", str(raw_output).upper())
    return list(dict.fromkeys(codes))


class _RemoteLlamaGuardEvalAgent(EvalAgent):
    """Remote response-only safety evaluator using a LlamaGuard endpoint.

    Local and remote evaluators share the same semantic input contract: only the
    current response text is evaluated. The original question is never sent.
    """

    def __init__(
        self,
        name: str,
        model_name: str,
        *,
        request_max_retries: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        logger=None,
    ) -> None:
        super().__init__(name=name, base_llm=None)
        self.model_name = model_name
        self.logger = logger
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_tokens = int(max_tokens)
        self.client = OpenAI(
            base_url=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8010/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", "local"),
            max_retries=int(request_max_retries),
        )

    def run(self, question_text: str, agent_response: str) -> str:
        # Protocol invariant: evaluate only the response text. question_text is kept
        # in the signature solely for compatibility with the existing orchestrator.
        del question_text
        response_text = str(agent_response)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": response_text},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        raw = completion.choices[0].message.content or ""
        verdict = _extract_llamaguard_verdict(raw)
        if self.logger is not None:
            self.logger.info("LlamaGuard raw output: %r", raw)
        if verdict is None:
            return raw
        return json.dumps(
            {
                "safety": verdict,
                "reason_codes": _extract_llamaguard_reason_codes(raw),
                "raw_llamaguard": raw,
            },
            ensure_ascii=False,
        )

def _parse_model_str(model_str: str):
    if not isinstance(model_str, str) or ":" not in model_str:
        raise ValueError("Model must use '<type>:<name>[:use_chat]' syntax")
    first = model_str.index(":")
    last = model_str.rindex(":")
    model_type = model_str[:first]
    if first == last:
        return model_type, model_str[first + 1 :], "true"
    return model_type, model_str[first + 1 : last], model_str[last + 1 :]


def _create_llm(
    model_str,
    prompt_template,
    logger,
    agent_label="",
    *,
    temperature,
    top_p,
    request_max_retries,
    do_sample,
    device=None,
    max_new_tokens=None,
    truncation=None,
    return_full_text=None,
    max_tokens=None,
):
    """Create one LLM using YAML-resolved generation settings only."""
    model_type, model_name, use_chat = _parse_model_str(model_str)
    if agent_label:
        logger.info("Loading %s model %r as %s", model_type, model_name, agent_label)

    if model_type == "huggingface":
        kwargs = {
            "model_name": model_name,
            "use_chat": use_chat,
            "prompt_template": prompt_template,
            "do_sample": bool(do_sample),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "device": device,
        }
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = int(max_new_tokens)
        if truncation is not None:
            kwargs["truncation"] = bool(truncation)
        if return_full_text is not None:
            kwargs["return_full_text"] = bool(return_full_text)
        return AnyHuggingFace(**kwargs)

    openai_compatible = {
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "fai": ("https://api.featherless.ai/v1", "FEATHERLESS_API_KEY"),
        "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    }
    if model_type == "openai" or model_type in openai_compatible:
        kwargs = {
            "model_name": model_name,
            "use_chat": use_chat,
            "prompt_template": prompt_template,
            "max_retries": int(request_max_retries),
        }
        if not (model_type == "openai" and "gpt-5" in model_name):
            kwargs.update(temperature=float(temperature), top_p=float(top_p))
        if model_type in openai_compatible:
            api_base, key_env = openai_compatible[model_type]
            kwargs.update(
                openai_api_base=api_base,
                openai_api_key=os.environ.get(key_env),
            )
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        return AnyOpenAILLM(**kwargs)

    if model_type == "google":
        return AnyGoogleGenerativeAI(
            model_name=model_name,
            use_chat=use_chat,
            prompt_template=prompt_template,
            temperature=float(temperature),
            top_p=float(top_p),
        )

    raise ValueError(
        f"Unsupported model type {model_type!r}; expected huggingface, openai, "
        "openrouter, fai, deepseek, or google"
    )


def init_agents(
    debater_models,
    evaluator_model,
    feedback_generator,
    logger,
    llamaguard_cuda_device: int,
    *,
    generation_config,
    peft_config,
    history_mode: str = "cardebate",
):
    """Initialize agents from fully resolved YAML-derived generation/PEFT config."""
    generation_config = dict(generation_config)
    peft_config = dict(peft_config)
    peft_memory = bool(peft_config["enabled"])
    debater_cfg = dict(generation_config["debater"])
    evaluator_cfg = dict(generation_config["evaluator"])
    feedback_cfg = dict(generation_config["feedback"])
    remote_guard_cfg = dict(generation_config["remote_llamaguard"])
    local_guard_cfg = dict(generation_config["local_llamaguard"])
    request_max_retries = int(generation_config["llm_request_max_retries"])
    answer_max_words = int(generation_config["answer_max_words"])
    format_repair_max_words = int(generation_config["format_repair_max_words"])

    debate_agents = []
    for index, model in enumerate(debater_models):
        model_type, _, _ = _parse_model_str(model)
        llm = _create_llm(
            model,
            debate_agent_prompt,
            logger,
            f"Agent-{index}",
            temperature=debater_cfg["temperature"],
            top_p=debater_cfg["top_p"],
            request_max_retries=request_max_retries,
            do_sample=debater_cfg["do_sample"],
            device=f"cuda:{index}",
            max_new_tokens=debater_cfg["max_new_tokens"],
            truncation=debater_cfg["truncation"],
            return_full_text=debater_cfg["return_full_text"],
            max_tokens=debater_cfg["max_tokens"],
        )
        agent_class = (
            PEFTDebateAgent
            if peft_memory and model_type == "huggingface"
            else DebateAgent
        )
        agent_kwargs = {
            "name": f"Agent-{index}",
            "base_llm": llm,
            "answer_max_words": answer_max_words,
            "format_repair_max_words": format_repair_max_words,
        }
        if agent_class is PEFTDebateAgent:
            agent_kwargs["peft_training_config"] = peft_config
        debate_agents.append(agent_class(**agent_kwargs))

    if evaluator_model is None:
        raise ValueError(f"{PROTOCOL_VERSION} requires evaluator_model")

    eval_type, eval_name, _ = _parse_model_str(evaluator_model)
    if eval_type == "openai" and eval_name == "moderation":
        eval_agent = EvalAgent(name="Agent-Eval", base_llm=None)
    elif eval_type == "openai" and (
        eval_name.strip().lower() == "evaluator"
        or _is_llamaguard_model(eval_name)
    ):
        eval_agent = _RemoteLlamaGuardEvalAgent(
            name="Agent-Eval",
            model_name=eval_name,
            request_max_retries=request_max_retries,
            temperature=remote_guard_cfg["temperature"],
            top_p=remote_guard_cfg["top_p"],
            max_tokens=remote_guard_cfg["max_tokens"],
            logger=logger,
        )
    elif eval_type == "huggingface" and _is_llamaguard_model(eval_name):
        eval_agent = EvalAgent(
            name="Agent-Eval",
            base_llm=LlamaGuard(
                model_name=eval_name,
                device=f"cuda:{llamaguard_cuda_device}",
                max_new_tokens=int(local_guard_cfg["max_new_tokens"]),
                pad_token_id=int(local_guard_cfg["pad_token_id"]),
            ),
        )
    else:
        eval_llm = _create_llm(
            evaluator_model,
            evaluator_prompt,
            logger,
            "Agent-Eval",
            temperature=evaluator_cfg["temperature"],
            top_p=evaluator_cfg["top_p"],
            request_max_retries=request_max_retries,
            do_sample=evaluator_cfg["do_sample"],
            device=f"cuda:{llamaguard_cuda_device}",
            max_new_tokens=evaluator_cfg["max_new_tokens"],
            truncation=evaluator_cfg["truncation"],
            return_full_text=evaluator_cfg["return_full_text"],
            max_tokens=evaluator_cfg["max_tokens"],
        )
        eval_agent = EvalAgent(name="Agent-Eval", base_llm=eval_llm)

    feedback_agent = None
    if feedback_generator is not None:
        feedback_llm = _create_llm(
            feedback_generator,
            (feedback_prompt_all_history if history_mode == "cardebate_all_history"
             else feedback_prompt),
            logger,
            "Agent-Feedback",
            temperature=feedback_cfg["temperature"],
            top_p=feedback_cfg["top_p"],
            request_max_retries=request_max_retries,
            do_sample=feedback_cfg["do_sample"],
            device=f"cuda:{len(debate_agents) + 1}",
            max_new_tokens=feedback_cfg["max_new_tokens"],
            truncation=feedback_cfg["truncation"],
            return_full_text=feedback_cfg["return_full_text"],
            max_tokens=feedback_cfg["max_tokens"],
        )
        feedback_agent = FeedbackAgent(
            name="Agent-Feedback", base_llm=feedback_llm
        )

    return debate_agents, eval_agent, feedback_agent

def _save_json(path: Path, value) -> None:
    """Atomically replace one JSON result/checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _save_memory_export(path: Path, long_term_memory: LongTermMemory) -> None:
    """Atomically export the current learned long-term memory as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    long_term_memory.save(str(tmp_path))
    os.replace(tmp_path, path)


def save_results_snapshot(
    results_dir: Path,
    debates: list[Debate],
    agent_stats: dict,
    long_term_memory: LongTermMemory,
    failed_questions: Optional[list[dict]] = None,
) -> dict:
    """Refresh all local ``results/`` artifacts from completed questions.

    This function is deliberately called after every completed question.  A run
    therefore always leaves a cumulative on-disk results snapshot instead of
    waiting for normal process termination.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    _save_memory_export(
        results_dir / "long_term_memory.jsonl", long_term_memory
    )
    _save_json(results_dir / "agent_reliability.json", agent_stats)

    failed_questions = list(failed_questions or [])
    metrics = calculate_debate_metrics(debates, failed_questions=failed_questions)
    _save_json(results_dir / "metrics.json", metrics)
    _save_json(results_dir / "failed_questions.json", failed_questions)
    _save_json(
        results_dir / "debate_history.json",
        [debate.serialize() for debate in debates],
    )
    return metrics


def save_debate_log(
    debate: Debate, checkpoint_folder: str, dataset_name: str, question_idx: int
) -> None:
    debate.save_to_json(
        str(Path(checkpoint_folder) / f"{dataset_name}_q{question_idx}.json")
    )


def save_checkpoint(checkpoint_folder: str, completed_debates: dict) -> None:
    _save_json(Path(checkpoint_folder) / "checkpoint.json", completed_debates)


def load_checkpoint(checkpoint_folder: str) -> dict:
    path = Path(checkpoint_folder) / "checkpoint.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError("checkpoint.json must contain an object")
    return value


def save_failed_questions(checkpoint_folder: str, failures: list[dict]) -> None:
    _save_json(Path(checkpoint_folder) / "failed_questions.json", failures)


def load_failed_questions(checkpoint_folder: str) -> list[dict]:
    path = Path(checkpoint_folder) / "failed_questions.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError("failed_questions.json must contain a list")
    return [dict(item) for item in value if isinstance(item, dict)]


def save_agent_stats(checkpoint_folder: str, agent_stats: dict) -> None:
    _save_json(Path(checkpoint_folder) / "agent_stats.json", agent_stats)


def load_agent_stats(checkpoint_folder: str) -> dict:
    path = Path(checkpoint_folder) / "agent_stats.json"
    if not path.exists():
        return default_agent_stats()
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_peft_training_data(checkpoint_folder: str, data: list) -> None:
    _save_json(Path(checkpoint_folder) / "peft_feedback.json", data)


def load_peft_training_data(checkpoint_folder: str) -> list:
    path = Path(checkpoint_folder) / "peft_feedback.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError("peft_feedback.json must contain a list")
    return value


def load_saved_debates(checkpoint_folder: str) -> list[Debate]:
    folder = Path(checkpoint_folder)
    if not folder.exists():
        return []
    debates = []
    for path in sorted(folder.glob("*_q*.json")):
        question_index = path.stem.rsplit("_q", 1)[-1]        
        if not question_index.isdigit():            
            continue
        debates.append(Debate.load_from_json(str(path)))
    return debates


def rebuild_agent_stats_from_debates(debates: list[Debate]) -> dict:
    stats = default_agent_stats()
    for debate in debates:
        initial = debate.state.initial
        record_r1_observations(
            stats,
            debate.get_question_text(),
            initial,
            question_embedding=None,
            use_expected_safety_mass=debate.use_expected_safety_mass,
            unsafe_sensitivity=debate.unsafe_sensitivity,
            safe_specificity=debate.safe_specificity,
            prior_floor=debate.evaluator_prior_floor,
            prior_ceiling=debate.evaluator_prior_ceiling,
        )
    return stats


RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run_manifest.json"


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _experiment_config_payload(
    *,
    debater_models,
    evaluator_model,
    datasets,
    feedback_generator,
    textual_memory_index,
    generation_config,
    peft_config,
    debate_config,
    abort_question_on_exhaustion,
    continue_experiment_after_question_abort,
    max_total_debates,
):
    """Return only fields that define the reproducible experiment condition."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "debater_models": list(debater_models),
        "evaluator_model": evaluator_model,
        "datasets": list(datasets),
        "feedback_generator": feedback_generator,
        "textual_memory_index": textual_memory_index,
        "vector_store_root": (
            str(
                Path(
                    os.environ.get("CHROMA_DB_ROOT") or "./chroma_db"
                ).expanduser().resolve()
            )
            if textual_memory_index
            else None
        ),
        "generation_config": dict(generation_config),
        "peft_config": dict(peft_config),
        "debate_config": dict(debate_config),
        "abort_question_on_exhaustion": bool(abort_question_on_exhaustion),
        "continue_experiment_after_question_abort": bool(
            continue_experiment_after_question_abort
        ),
        "max_total_debates": max_total_debates,
    }


def _experiment_config_fingerprint(payload) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _checkpoint_contains_run_state(checkpoint_path: Path) -> bool:
    known_files = {
        "checkpoint.json",
        "agent_stats.json",
        "failed_questions.json",
        "peft_feedback.json",
        "long_term_memory.jsonl",
    }
    if any((checkpoint_path / name).exists() for name in known_files):
        return True
    return any(checkpoint_path.glob("*_q*.json"))


def _ensure_run_manifest(
    checkpoint_path: Path,
    experiment_payload: dict,
    logger,
) -> dict:
    """Bind a checkpoint directory to exactly one resolved experiment config."""
    fingerprint = _experiment_config_fingerprint(experiment_payload)
    manifest_path = checkpoint_path / RUN_MANIFEST_FILENAME
    expected = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "config_fingerprint": fingerprint,
        "experiment_config": experiment_payload,
    }

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if not isinstance(existing, dict):
            raise ValueError(f"Invalid run manifest: {manifest_path}")
        if int(existing.get("schema_version", 0) or 0) != RUN_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Checkpoint run_manifest.json has an unsupported schema; "
                "start a new checkpoint directory or migrate it explicitly"
            )
        if existing.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(
                "Checkpoint run manifest belongs to a different protocol version"
            )
        saved_fingerprint = str(existing.get("config_fingerprint") or "")
        if saved_fingerprint != fingerprint:
            raise ValueError(
                "Checkpoint configuration does not match the current YAML-resolved "
                "experiment configuration. Refusing to mix experimental conditions. "
                "Use a new checkpoint directory. "
                f"saved={saved_fingerprint or '<missing>'} current={fingerprint}"
            )
        logger.info("Verified checkpoint config fingerprint: %s", fingerprint)
        return existing

    if _checkpoint_contains_run_state(checkpoint_path):
        raise ValueError(
            "Checkpoint directory contains prior run state but no run_manifest.json. "
            "Strict YAML-single-source mode will not resume an unbound checkpoint. "
            "Use a new checkpoint directory or migrate the old run explicitly."
        )

    _save_json(manifest_path, expected)
    logger.info("Created checkpoint config fingerprint: %s", fingerprint)
    return expected


def run_debate(
    debater_models,
    evaluator_model,
    datasets,
    output_file,
    *,
    feedback_generator,
    textual_memory_index,
    generation_config,
    peft_config,
    debate_config,
    abort_question_on_exhaustion,
    continue_experiment_after_question_abort,
    llamaguard_cuda_device,
    model_output_dir=None,
    model_output_files=None,
    checkpoint_dir=None,
    max_total_debates=None,
):
    """Run three answer rounds with evaluator-triggered embedded critique."""
    logger = setup_logger(output_file)

    # Resolve the run checkpoint root first. The launcher already places checkpoints
    # under RUN_ROOT/checkpoints, so this is the most reliable anchor for deriving
    # the sibling RUN_ROOT/model_outputs directory.
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    checkpoint_dir = checkpoint_dir or f"checkpoints/debate_{timestamp}"
    checkpoint_path = Path(checkpoint_dir).expanduser().resolve()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(checkpoint_path)

    output_path = Path(output_file).expanduser().resolve()

    # Resolution priority:
    #   1. explicit --model_output_dir / Python argument
    #   2. MODEL_OUTPUT_DIR environment variable
    #   3. sibling of the resolved checkpoint directory
    #   4. output_file-derived fallback
    explicit_model_output_dir = model_output_dir
    if explicit_model_output_dir is None:
        explicit_model_output_dir = os.environ.get("MODEL_OUTPUT_DIR")

    if explicit_model_output_dir:
        model_output_path = Path(explicit_model_output_dir).expanduser().resolve()
    elif checkpoint_path.name == "checkpoints":
        model_output_path = checkpoint_path.parent / "model_outputs"
    elif checkpoint_path.parent.name == "checkpoints":
        # Also support checkpoint layouts such as RUN_ROOT/checkpoints/debate_xxx.
        model_output_path = checkpoint_path.parent.parent / "model_outputs"
    elif output_path.parent.name == "logs":
        model_output_path = output_path.parent.parent / "model_outputs"
    else:
        model_output_path = output_path.parent / "model_outputs"

    model_output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Resolved checkpoint directory: %s", checkpoint_path)
    logger.info("Per-model JSONL directory: %s", model_output_path)

    resolved_model_output_files = _resolve_model_output_files(
        model_output_files,
        debater_models,
        evaluator_model,
        feedback_generator,
    )
    logger.info(
        "Resolved model output files: %s",
        json.dumps(resolved_model_output_files, ensure_ascii=False, sort_keys=True),
    )

    generation_config = dict(generation_config)
    peft_config = dict(peft_config)
    debate_config = dict(debate_config)

    peft_memory = bool(peft_config["enabled"])
    train_steps = int(peft_config["train_steps"])
    peft_directory = str(peft_config["directory"])
    if train_steps <= 0:
        raise ValueError("peft_config.train_steps must be positive")

    if not isinstance(debate_config["format_retry_reasons"], list):
        raise TypeError("debate_config.format_retry_reasons must be a list")
    if int(debate_config["format_retry_max_retries"]) < 0:
        raise ValueError("debate_config.format_retry_max_retries must be >= 0")

    logger.info(
        "CaRDebate history mode: %s (retrieval=%s, feedback=%s, successful_only=%s)",
        debate_config.get("history_mode", "cardebate"),
        debate_config.get("history_enabled", True),
        debate_config.get("feedback_enabled", True),
        debate_config["successful_history_only"],
    )

    experiment_payload = _experiment_config_payload(
        debater_models=debater_models,
        evaluator_model=evaluator_model,
        datasets=datasets,
        feedback_generator=feedback_generator,
        textual_memory_index=textual_memory_index,
        generation_config=generation_config,
        peft_config=peft_config,
        debate_config=debate_config,
        abort_question_on_exhaustion=abort_question_on_exhaustion,
        continue_experiment_after_question_abort=continue_experiment_after_question_abort,
        max_total_debates=max_total_debates,
    )
    run_manifest = _ensure_run_manifest(
        checkpoint_path, experiment_payload, logger
    )

    # Prevent different ablation modes from sharing learned memory records.
    history_mode = str(debate_config.get("history_mode", "cardebate"))
    effective_memory_index = textual_memory_index
    if history_mode == "cardebate_all_history" and effective_memory_index:
        effective_memory_index = str(effective_memory_index)
        if not effective_memory_index.endswith("-all-history"):
            effective_memory_index += "-all-history"

    elif history_mode == "cardebate_without_rollback" and effective_memory_index:
        effective_memory_index = str(effective_memory_index)
        if not effective_memory_index.endswith("-without-rollback"):
            effective_memory_index += "-without-rollback"

    elif history_mode == "cardebate_no_history":
        effective_memory_index = None

    if effective_memory_index:
        long_term_memory: LongTermMemory = VectorStoreMemory(
            description="Long-Term Memory",
            index_name=str(effective_memory_index),
            allow_all_history=not bool(debate_config["successful_history_only"]),
        )
        memory_checkpoint_path = None
    else:
        long_term_memory = LongTermMemory(
            "Long-Term Memory",
            allow_all_history=not bool(debate_config["successful_history_only"]),
        )
        memory_checkpoint_path = checkpoint_path / "long_term_memory.jsonl"
        if memory_checkpoint_path.exists():
            long_term_memory.load(str(memory_checkpoint_path))

    dataset_objects = load_datasets(datasets)
    active_feedback_generator = (
        feedback_generator if debate_config.get("feedback_enabled", True) else None
    )
    debate_agents, eval_agent, feedback_agent = init_agents(
        debater_models,
        evaluator_model,
        active_feedback_generator,
        logger,
        int(llamaguard_cuda_device),
        generation_config=generation_config,
        peft_config=peft_config,
        history_mode=history_mode,
    )
    if not debate_config.get("feedback_enabled", True):
        feedback_agent = None

    completed_debates = load_checkpoint(checkpoint_dir)
    debates = load_saved_debates(checkpoint_dir)
    
    for debate in debates:    
        debate.long_term_memory = long_term_memory

    failed_questions = load_failed_questions(checkpoint_dir)

    if debates and memory_checkpoint_path is not None and not memory_checkpoint_path.exists():
        raise ValueError(
            "Found debate checkpoints but no array long-term-memory checkpoint. "
            "Refusing to resume with silently lost learned memory."
        )

    agent_stats = load_agent_stats(checkpoint_dir)
    if debates and (
        not (agent_stats or {}).get("agents")
        or agent_stats_need_rebuild(agent_stats)
    ):
        logger.info("Rebuilding Bayesian R1 history from current debate checkpoints")
        agent_stats = rebuild_agent_stats_from_debates(debates)
        save_agent_stats(checkpoint_dir, agent_stats)
    elif not debates and agent_stats_need_rebuild(agent_stats):
        agent_stats = default_agent_stats()

    peft_feedback = load_peft_training_data(checkpoint_dir)
    peft_directory = peft_directory or ""
    hf_cache = os.environ.get("HF_HUB_CACHE", "")

    def maybe_train_peft() -> None:
        if not peft_memory:
            return
        if len(peft_feedback) < int(train_steps):
            return
        if len(peft_feedback) % int(train_steps) != 0:
            return
        logger.info(
            "Triggering PEFT training with %d accepted feedback rules",
            len(peft_feedback),
        )
        for agent in debate_agents:
            if not isinstance(agent, PEFTDebateAgent):
                continue
            model_name = str(agent.base_llm.model_name).strip("/")
            save_dir = Path(hf_cache) / peft_directory / model_name / f"{agent.name}_LoRA"
            save_dir.mkdir(parents=True, exist_ok=True)
            agent.trainPEFT(peft_feedback, str(save_dir))

    max_total = (
        float("inf") if max_total_debates is None else int(max_total_debates)
    )

    # Local results are cumulative and refreshed after every completed question.
    # Prefer the launcher's explicit RESULTS_DIR; otherwise derive a sibling of
    # RUN_ROOT/checkpoints when that conventional layout is used.
    if os.environ.get("RESULTS_DIR"):
        results_dir = Path(os.environ["RESULTS_DIR"]).expanduser().resolve()
    elif checkpoint_path.name == "checkpoints":
        results_dir = checkpoint_path.parent / "results"
    elif checkpoint_path.parent.name == "checkpoints":
        results_dir = checkpoint_path.parent.parent / "results"
    else:
        results_dir = checkpoint_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Resolved results directory: %s", results_dir)

    debate_error = None

    try:
        stop = False
        for dataset_name, dataset in dataset_objects.items():
            last_processed = int(completed_debates.get(dataset_name, -1))
            failed_indices = set()
            for failure in failed_questions:
                if str(failure.get("dataset")) != str(dataset_name):
                    continue
                try:
                    failed_indices.add(int(failure.get("question_index", -1)))
                except (TypeError, ValueError):
                    continue
            for idx, question in enumerate(dataset):
                if idx <= last_processed and idx not in failed_indices:
                    continue
                if len(debates) >= max_total and idx not in failed_indices:
                    stop = True
                    break
                if not isinstance(question, dict) or not str(question.get("text", "")).strip():
                    raise ValueError(
                        f"Dataset {dataset_name!r} question {idx} lacks non-empty text"
                    )

                long_term_memory.set_seed_memories(
                    question.get("memory", []), category=question.get("label")
                )

                debate = Debate(
                    question,
                    debate_agents,
                    eval_agent,
                    feedback_agent,
                    long_term_memory,
                    log_file=output_file,
                    model_output_dir=str(model_output_path),
                    model_output_files=resolved_model_output_files,
                    dataset_name=str(dataset_name),
                    question_index=int(idx),
                    agent_stats=agent_stats,
                    answer_max_words=int(generation_config["answer_max_words"]),
                    **debate_config,
                )
                try:
                    debate.start(rounds=3)
                except AnswerGenerationExhausted as exc:
                    if (
                        not abort_question_on_exhaustion
                        or not continue_experiment_after_question_abort
                    ):
                        raise
                    failure = {
                        "dataset": str(dataset_name),
                        "question_index": int(idx),
                        "category": question.get("label"),
                        "question": str(question.get("text", "")),
                        "error_type": type(exc).__name__,
                        "stage": getattr(exc, "stage", None),
                        "agent": getattr(exc, "agent", None),
                        "failure_reason": getattr(exc, "failure_reason", None),
                        "message": str(exc),
                        "audit": getattr(exc, "audit", {}),
                    }
                    failed_questions = [
                        item for item in failed_questions
                        if not (
                            str(item.get("dataset")) == str(dataset_name)
                            and int(item.get("question_index", -1)) == int(idx)
                        )
                    ]
                    failed_questions.append(failure)
                    logger.error(
                        "Question %s/%d aborted after structured-format retry exhaustion "
                        "(stage=%s agent=%s reason=%s); continuing experiment",
                        dataset_name, idx, failure.get("stage"), failure.get("agent"),
                        failure.get("failure_reason"),
                    )
                    completed_debates[dataset_name] = max(
                        int(completed_debates.get(dataset_name, -1)), int(idx)
                    )
                    save_checkpoint(checkpoint_dir, completed_debates)
                    save_failed_questions(checkpoint_dir, failed_questions)
                    save_agent_stats(checkpoint_dir, agent_stats)
                    if memory_checkpoint_path is not None:
                        long_term_memory.save(str(memory_checkpoint_path))
                    save_results_snapshot(
                        results_dir, debates, agent_stats, long_term_memory, failed_questions
                    )
                    continue

                debates.append(debate)

                accepted_rules = debate.get_stored_feedback_rules()
                if peft_memory and accepted_rules:
                    peft_feedback.extend({"text": rule} for rule in accepted_rules)
                    # Persist accepted PEFT data immediately, independent of when
                    # the training cadence fires.
                    save_peft_training_data(checkpoint_dir, peft_feedback)
                    maybe_train_peft()

                save_debate_log(debate, checkpoint_dir, dataset_name, idx)
                failed_questions = [
                    item for item in failed_questions
                    if not (
                        str(item.get("dataset")) == str(dataset_name)
                        and int(item.get("question_index", -1)) == int(idx)
                    )
                ]
                completed_debates[dataset_name] = max(
                    int(completed_debates.get(dataset_name, -1)), int(idx)
                )
                save_checkpoint(checkpoint_dir, completed_debates)
                save_failed_questions(checkpoint_dir, failed_questions)
                save_agent_stats(checkpoint_dir, agent_stats)
                if memory_checkpoint_path is not None:
                    long_term_memory.save(str(memory_checkpoint_path))

                # Results are cumulative. Refresh them only after this question has
                # fully completed and its checkpoint/history state is durable.
                save_results_snapshot(
                    results_dir, debates, agent_stats, long_term_memory, failed_questions
                )
                logger.info(
                    "Results snapshot updated after %s question %d (%d debates total)",
                    dataset_name,
                    idx,
                    len(debates),
                )

            if stop:
                break
    except Exception as exc:
        logger.exception("Debate run failed: %s", exc)
        debate_error = exc
    finally:
        # Refresh once more on normal exit or failure. Completed questions were
        # already flushed individually, so this is only a final consistency pass.
        try:
            metrics = save_results_snapshot(
                results_dir, debates, agent_stats, long_term_memory, failed_questions
            )
        except Exception as results_exc:
            logger.exception("Final results snapshot failed: %s", results_exc)
            metrics = calculate_debate_metrics(debates, failed_questions=failed_questions)
            if debate_error is None:
                debate_error = results_exc

        experiment_parameters = {
            "protocol_version": PROTOCOL_VERSION,
            "round_structure": {
                "round1": "independent_answer->evaluate->critique",
                "round2": "revision->evaluate->critique",
                "round3": "final_revision->evaluate->feedback",
            },
            "debater_models": debater_models,
            "evaluator_model": evaluator_model,
            "feedback_generator": feedback_generator,
            "datasets": datasets,
            "model_output_dir": str(model_output_path),
            "model_output_files": resolved_model_output_files,
            "memory_backend": "chroma" if textual_memory_index else "array",
            "generation_config": generation_config,
            "peft_config": peft_config,
            "debate_config": debate_config,
            "abort_question_on_exhaustion": bool(abort_question_on_exhaustion),
            "continue_experiment_after_question_abort": bool(
                continue_experiment_after_question_abort
            ),
            "max_total_debates": max_total_debates,
            "config_fingerprint": run_manifest["config_fingerprint"],
            "run_id": timestamp,
        }
        try:
            experiment_parameters["commit_hash"] = (
                subprocess.check_output(["git", "rev-parse", "HEAD"])
                .decode("utf-8")
                .strip()
            )
        except Exception:
            pass

        try:
            log_results_to_swanlab(
                metrics, experiment_parameters, debates, str(results_dir)
            )
        except Exception as exc:
            logger.warning("SwanLab logging skipped/failed: %s", exc)

    if debate_error is not None:
        raise debate_error
    return debates
