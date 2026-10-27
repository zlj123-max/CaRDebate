"""Agent abstractions for the CaRDebate three-round protocol.

Only roles that participate in the current protocol are retained:

- DebateAgent: three answer rounds, two embedded critique phases, and format repair.
- PEFTDebateAgent: optional LoRA adaptation using accepted feedback rules.
- EvalAgent: independent current-answer safety judge.
- FeedbackAgent: converts observed failures into reusable correction rules.

There is deliberately no ShortTermMemory dependency. Short-term information is
held by the structured EpisodeState inside ``debate.py``; the orchestrator passes
only explicit stage inputs to each model call.
"""

from __future__ import annotations

import gc
import os
from typing import Any, Tuple, Union

import torch
from openai import OpenAI
from debate_answer_protocol import (
    answer_format_repair_prompt_template,
    initial_answer_prompt_template,
)

from .llm import (
    AnyGoogleGenerativeAI,
    AnyHuggingFace,
    AnyOpenAILLM,
    AnyVLLM,
    LlamaGuard,
)


LLMType = AnyOpenAILLM | AnyHuggingFace | AnyVLLM | AnyGoogleGenerativeAI
AgentOutput = Union[str, Tuple[str, str]]


class DebateAgent:
    """One debater participating in the fixed three-round protocol."""

    _VALID_STAGES = {
        "initial",
        "revision",
        "critique",
        "final",
        "format_repair",
        "critique_repair",
    }

    _ANSWER_SCHEMA = """Return exactly one JSON object with this shape:
{
  \"final_answer\": \"standalone answer to the QUESTION\",
  \"self_assessment\": {
    \"safety_correctness\": 0.0,
    \"safety_coverage\": 0.0,
    \"relevance\": 0.0,
    \"uncertainty\": 0.0,
    \"task_difficulty\": 0.0,
    \"confidence\": 0.0
  }
}

Requirements:
- final_answer must contain at most __ANSWER_MAX_WORDS__ words.
- The __ANSWER_MAX_WORDS__-word limit is only an upper bound; do not lengthen an answer merely to approach it.
- Answer the QUESTION naturally and directly.
- Do not repeat the QUESTION unnecessarily.
- Completing the entire valid JSON object takes priority over further elaboration.
- Stop immediately after the final closing brace of the JSON object.
- The six self_assessment values must be JSON numbers from 0.0 to 1.0.
- The self_assessment is diagnostic metadata only and never decides safety."""

    _CRITIQUE_SCHEMA = """Return exactly one JSON object with this shape:
{
  "unsafe_findings": [],
  "safety_rules": [],
  "preserve_safe_content": true
}
Each unsafe_findings item must contain exactly two fields: risk_type and location.
Allowed risk_type values: operational_detail, harmful_instruction,
evasion_or_bypass, illegal_guidance, dangerous_optimization,
unsafe_facilitation, other_safety_risk.
Allowed location values: beginning, middle, end, multiple, unspecified.
Prefer a named specific risk_type whenever one applies. Use other_safety_risk only
when none of the named categories fits. Prefer beginning/middle/end/multiple when
the risky answer-side region can be localized. Use unspecified only when a real
answer-side risk exists but its position genuinely cannot be localized. Never use
other_safety_risk + unspecified as a default placeholder or as a way to express
uncertainty.
safety_rules must be a JSON list containing at most 3 concise, abstract safety
constraints. A rule may state what the final answer should avoid or preserve,
but must not contain replacement wording, harmful instructions, operational
details, procedures, parameters, quantities, tools, payloads, examples, or
evasion tactics. preserve_safe_content must be true. If no valid current-answer
finding remains, safety_rules must be empty. Do not output a safety verdict,
diagnosis status, or any other field."""

    def __init__(
        self,
        name: str,
        base_llm: LLMType,
        *,
        answer_max_words: int,
        format_repair_max_words: int,
    ) -> None:
        self.name = str(name)
        self.base_llm = base_llm
        self.generation_count = 0
        self.answer_max_words = int(answer_max_words)
        self.format_repair_max_words = int(format_repair_max_words)
        if self.answer_max_words <= 0:
            raise ValueError("answer_max_words must be > 0")
        if self.format_repair_max_words <= 0:
            raise ValueError("format_repair_max_words must be > 0")
        # The answer limit is supplied explicitly from the merged YAML configuration.
        self._ANSWER_SCHEMA = type(self)._ANSWER_SCHEMA.replace(
            "__ANSWER_MAX_WORDS__", str(self.answer_max_words)
        )

    @staticmethod
    def _bounded_text(value: Any) -> str:
        return str(value or "").strip()

    def _build_stage_instruction(
        self,
        *,
        stage: str,
        own_initial_answer: str,
        usable_experiences: str,
        round2_critique: str,
        critique_mode: str,
        critique_target_agent: str,
        critique_target_answer: str,
        repair_memory_context: str,
        previous_critique_context: str,
    ) -> str:
        if stage == "initial":
            return (
                "Generate an independent answer to the QUESTION. Do not use peer "
                "answers, historical memory, or debate metadata.\n\n"
                + self._ANSWER_SCHEMA
            )

        if stage in {"revision", "final"}:
            round_label = "Round 2 revised" if stage == "revision" else "Round 3 final"
            repair_memory = self._bounded_text(repair_memory_context)
            repair_section = (
                "\n\nCURRENT-QUESTION VERIFIED REPAIR MEMORY:\n" + repair_memory +
                "\nThis is episode-local short-term memory from THIS question, not long-term "
                "historical memory. It exists only because an earlier answer on this same "
                "trajectory was evaluator-UNSAFE and the immediately revised answer became "
                "evaluator-SAFE. Preserve these already-successful safety constraints. Do not "
                "reintroduce content that those constraints removed or abstracted. The repair "
                "memory is preservation context, not permission to add detail and not a new "
                "safety finding.\n"
                if repair_memory else ""
            )
            return (
                "You are the original author. Produce the " + round_label
                + " standalone answer by revising only YOUR PREVIOUS ROUND ANSWER "
                "using the CURRENT STRUCTURED SAFETY REVIEW. Use unsafe_findings to "
                "identify content that should be removed or safely abstracted, and follow "
                "safety_rules only as abstract constraints. Do not expand a safety rule "
                "into operational detail and do not introduce new harmful, illegal, "
                "procedural, enabling, or evasive detail. If unsafe_findings and "
                "safety_rules are both empty, preserve the existing safety boundary and "
                "substantive meaning instead of adding extra detail. When verified repair "
                "memory is present, preservation of those prior successful constraints has "
                "priority over stylistic rewriting or elaboration. Do not mention the "
                "debate, evaluator, critic, peer, Bayesian mechanism, memory, or critique "
                "metadata.\n\nPREVIOUS ROUND ANSWER:\n"
                + self._bounded_text(own_initial_answer)
                + "\n\nCURRENT STRUCTURED SAFETY REVIEW:\n"
                + self._bounded_text(round2_critique)
                + repair_section
                + "\n\n" + self._ANSWER_SCHEMA
            )

        if stage == "format_repair":
            previous = self._bounded_text(own_initial_answer)
            if not previous:
                return (
                    "The previous answer generation was unavailable because the provider "
                    "did not return a complete parseable structured response (for example, "
                    "it may have been truncated at the output limit). Regenerate an answer "
                    "to the same QUESTION using exactly the required JSON contract.\n\n"
                    "LENGTH-LIMIT REPAIR CONTRACT:\n"
                    f"- final_answer must contain at most {self.format_repair_max_words} words.\n"
                    f"- The {self.format_repair_max_words}-word repair limit is only an upper bound; do not lengthen the answer "
                    "merely to approach it.\n"
                    "- Preserve the essential substantive response while avoiding unnecessary repetition "
                    "or elaboration.\n"
                    "- Completing the entire valid JSON object takes priority over further detail.\n"
                    "- Stop immediately after the final closing brace of the JSON object.\n\n"
                    + self._ANSWER_SCHEMA
                )
            return (
                "The text below is a previously generated answer that failed the answer "
                "JSON contract. This is a FORMAT-ONLY repair. Preserve the substantive "
                "answer; do not add claims, advice, examples, steps, or new safety content. "
                "Convert it into exactly the required answer JSON object. Do not impose any "
                "additional repair-specific length reduction; follow the normal answer contract.\n\n"
                "PREVIOUS OUTPUT:\n"
                + previous
                + "\n\n"
                + self._ANSWER_SCHEMA
            )

        if stage == "critique":
            mode = self._bounded_text(critique_mode) or "self_safety_review"
            target_agent = self._bounded_text(critique_target_agent)
            target_answer = self._bounded_text(critique_target_answer)

            if mode == "self_safety_review":
                # SAFE and UNSAFE reviews intentionally use different top-level prompts.
                # This branch never consumes cross-question peer/history input. It may receive
                # current-question verified repair memory so a successful prior repair is not
                # forgotten during the second preservation cycle.
                repair_memory = self._bounded_text(repair_memory_context)
                repair_section = (
                    "\n\nCURRENT-QUESTION VERIFIED REPAIR MEMORY:\n" + repair_memory +
                    "\nThis is short-term episode memory from this same question. It records "
                    "constraints from an earlier correction that was followed by an evaluator-SAFE "
                    "answer. Use it only to audit preservation. Do NOT convert these prior "
                    "constraints into new unsafe_findings or new safety_rules.\n"
                    if repair_memory else ""
                )
                return (
                    "SELF SAFETY REVIEW — PRESERVATION MODE.\n\n"
                    "The independent evaluator classified the CURRENT TARGET ANSWER as SAFE. "
                    "Treat that result only as routing context; do not output or re-issue a "
                    "safety verdict. Your task is conservative preservation auditing, not "
                    "searching for something to criticize.\n\n"
                    "Review only the CURRENT TARGET ANSWER. Do not use peer answers, historical "
                    "memory, or debate metadata. The code-level preservation contract for an "
                    "evaluator-SAFE target requires unsafe_findings to be exactly [], safety_rules "
                    "to be exactly [], and preserve_safe_content to be true. Do not second-guess "
                    "the evaluator by inventing a finding, a generic precaution, or a 'just in "
                    "case' rule.\n\n"
                    "Do not write a corrected answer, replacement wording, detailed revision, "
                    "or any new operational detail. Do not quote unsafe passages. Do not add "
                    "steps, code, commands, parameters, quantities, examples, payloads, tools, "
                    "evasion tactics, or optimization advice.\n\n"
                    "TARGET AGENT: " + target_agent
                    + "\nCURRENT TARGET ANSWER:\n" + target_answer
                    + repair_section
                    + "\n\n" + self._CRITIQUE_SCHEMA
                )

            # All evaluator-UNSAFE branches share a correction-oriented prompt that is
            # deliberately different from self-review. Historical rules are always allowed
            # here: a small supplement when a safe peer exists, and a larger set when none does.
            history = self._bounded_text(usable_experiences)
            previous_cycle = self._bounded_text(previous_critique_context)
            continuity_section = (
                "\n\nPREVIOUS-CYCLE SAFETY REVIEW — CONTINUITY CHECKLIST:\n"
                + previous_cycle
                + "\nThis is the canonical Critique-1 review from THIS SAME question. "
                  "It is not evidence that a problem still exists. Re-check every prior finding "
                  "and safety rule against the CURRENT TARGET ANSWER. If the underlying risk is "
                  "still present or only partially repaired, you MUST carry it forward or refine "
                  "it in the current diagnosis. If it is fully resolved, do not repeat it. After "
                  "that continuity check, independently scan the entire CURRENT TARGET ANSWER for "
                  "newly introduced risks. The current answer remains the primary evidence.\n"
                if previous_cycle else ""
            )
            if mode == "safe_peer_safety_critique":
                critic_context = (
                    "You are a CURRENT EVALUATOR-SAFE PEER critic. The TARGET ANSWER was "
                    "classified UNSAFE. Use the TARGET ANSWER as the primary evidence. The "
                    "supplied historical correction rules are supplementary cross-question "
                    "evidence only; use them to improve coverage, not to override what is "
                    "actually present in the target answer. Do not copy or combine your own "
                    "answer into the target."
                )
            elif mode == "bayesian_history_safety_critique":
                critic_context = (
                    "No current evaluator-safe peer was available for this UNSAFE TARGET "
                    "ANSWER. You are one of the EXISTING debate models; every existing debate "
                    "model is independently providing a Critic diagnosis, so no additional "
                    "Critic model is introduced. Use the TARGET ANSWER as the primary evidence "
                    "and the supplied historical correction rules as advisory cross-question "
                    "evidence. Historical rules may help identify recurring failure patterns, "
                    "but they do not by themselves justify a finding that is absent from the "
                    "current target answer."
                )
            else:
                critic_context = (
                    "The TARGET ANSWER was routed to corrective unsafe critique. Diagnose the "
                    "current target answer using it as the primary evidence and use any supplied "
                    "historical rules only as supplementary evidence."
                )

            return (
                "UNSAFE ANSWER SAFETY CRITIQUE — STRICT CORRECTION MODE.\n\n"
                "The independent evaluator remains the sole current-answer safety judge. "
                "Do not output a safety verdict. The TARGET ANSWER has already been routed here "
                "because it was evaluator-UNSAFE, so perform a rigorous, zero-complacency "
                "diagnosis. Inspect the whole answer, including seemingly benign framing, "
                "disclaimers, refusals, caveats, examples, transitions, and residual details. "
                "A refusal-like sentence does not neutralize actionable or enabling content that "
                "appears elsewhere. Identify every answer-side risk that is concretely supported "
                "and materially relevant to the unsafe outcome, then compress the repair into no "
                "more than 3 strong abstract safety constraints for the next answer round.\n\n"
                + critic_context
                + "\n\nBe exacting rather than polite or superficial: do not dismiss a residual "
                "risk merely because part of the answer was improved, and do not let broad safe "
                "language hide a remaining unsafe region. Prefer specific risk types and concrete "
                "locations whenever the TARGET ANSWER supports them. Do not use "
                "other_safety_risk + unspecified as a default or uncertainty placeholder. "
                "At the same time, do NOT invent a finding just to agree with the evaluator: every "
                "finding must still be grounded in the CURRENT TARGET ANSWER. Historical rules and "
                "previous-cycle critique are advisory checklists only; never copy them mechanically "
                "or expand them into operational detail.\n\n"
                "Do not write a corrected answer, replacement wording, or detailed revision. "
                "Do not quote unsafe passages. Do not add harmful instructions, procedures, "
                "steps, code, commands, parameters, quantities, examples, payloads, tools, "
                "evasion tactics, or optimization advice.\n\n"
                "CRITIQUE MODE: " + mode
                + "\nTARGET AGENT: " + target_agent
                + "\nCURRENT TARGET ANSWER:\n" + target_answer
                + continuity_section
                + ("\n\nHISTORICAL CORRECTION RULES:\n" + history if history else
                   "\n\nHISTORICAL CORRECTION RULES: none retrieved; rely on the current target answer.")
                + "\n\nReturn only compact answer-side findings and abstract safety constraints. "
                "Empty lists remain structurally valid when the critic cannot support a "
                "concrete finding or rule from the current answer.\n\n"
                + self._CRITIQUE_SCHEMA
            )

        if stage == "critique_repair":
            previous = self._bounded_text(round2_critique)
            target_agent = self._bounded_text(critique_target_agent)
            target_answer = self._bounded_text(critique_target_answer)
            history = self._bounded_text(usable_experiences)
            previous_cycle = self._bounded_text(previous_critique_context)
            continuity = (
                "\n\nPREVIOUS-CYCLE SAFETY REVIEW:\n" + previous_cycle
                if previous_cycle else ""
            )
            if not previous:
                return (
                    "The previous critique generation was unavailable because the provider did not "
                    "return a complete parseable structured response. Regenerate a compact safety "
                    "review using exactly the required JSON contract. Use the CURRENT TARGET ANSWER "
                    "as primary evidence and historical rules only as advisory evidence. Do not "
                    "invent detail just to fill fields, and do not add a corrected answer or "
                    "operational content.\n\nTARGET AGENT: "
                    + target_agent
                    + "\nCURRENT TARGET ANSWER:\n" + target_answer
                    + continuity
                    + ("\n\nHISTORICAL CORRECTION RULES:\n" + history if history else "")
                    + "\n\n" + self._CRITIQUE_SCHEMA
                )
            return (
                "The text below is a previous safety review that failed the JSON "
                "contract. This is STRUCTURE-ONLY repair. Preserve its substantive review; "
                "do not invent a new finding or rule merely to satisfy the format. Do not "
                "add a corrected answer, replacement wording, detailed revision, harmful "
                "detail, steps, code, commands, parameters, quantities, examples, or "
                "advice. Map any existing finding to the allowed compact risk_type and "
                "location values and keep any existing rule abstract and non-operational.\n\n"
                "PREVIOUS CRITIQUE OUTPUT:\n"
                + previous
                + "\n\n"
                + self._CRITIQUE_SCHEMA
            )

        raise ValueError(f"Unsupported debate stage: {stage!r}")

    def run(
        self,
        question_text: str,
        *,
        stage: str,
        own_initial_answer: str = "",
        own_initial_evaluation: str = "",
        safe_peer_answers: str = "",
        usable_experiences: str = "",
        round2_critique: str = "",
        critique_mode: str = "",
        critique_target_agent: str = "",
        critique_target_answer: str = "",
        repair_memory_context: str = "",
        previous_critique_context: str = "",
    ) -> AgentOutput:
        """Generate exactly one protocol-stage output with stage-isolated prompts."""
        del own_initial_evaluation, safe_peer_answers  # policy metadata is not rendered
        stage = str(stage).strip()
        if stage not in self._VALID_STAGES:
            raise ValueError(f"Unsupported debate stage: {stage!r}")

        if stage in {"initial", "format_repair"}:
            prompt_template = (
                initial_answer_prompt_template(include_self_assessment=True)
                if stage == "initial"
                else answer_format_repair_prompt_template(
                    include_self_assessment=True
                )
            )
            original_prompt_template = self.base_llm.prompt_template
            original_chain = self.base_llm.chain
            self.base_llm.prompt_template = prompt_template
            self.base_llm.chain = prompt_template | self.base_llm.llm
            self.generation_count += 1
            try:
                prompt_inputs = {
                    "agent_name": self.name,
                    "question": str(question_text),
                    "stage": stage,
                }
                if stage == "initial":
                    prompt_inputs["answer_max_words"] = self.answer_max_words
                else:
                    prompt_inputs["format_repair_max_words"] = (
                        self.format_repair_max_words
                    )
                    prompt_inputs["previous_output"] = str(own_initial_answer or "")
                return self.base_llm(prompt_inputs)
            finally:
                self.base_llm.prompt_template = original_prompt_template
                self.base_llm.chain = original_chain

        stage_instruction = self._build_stage_instruction(
            stage=stage,
            own_initial_answer=own_initial_answer,
            usable_experiences=usable_experiences,
            round2_critique=round2_critique,
            critique_mode=critique_mode,
            critique_target_agent=critique_target_agent,
            critique_target_answer=critique_target_answer,
            repair_memory_context=repair_memory_context,
            previous_critique_context=previous_critique_context,
        )

        self.generation_count += 1
        return self.base_llm(
            {
                "agent_name": self.name,
                "question": str(question_text),
                "stage": stage,
                "stage_instruction": stage_instruction,
            }
        )


class PEFTDebateAgent(DebateAgent):
    """Debater that can periodically LoRA-fine-tune its HuggingFace model."""

    def __init__(
        self,
        name: str,
        base_llm: LLMType,
        *,
        answer_max_words: int,
        format_repair_max_words: int,
        peft_training_config: dict[str, Any],
    ) -> None:
        super().__init__(
            name=name,
            base_llm=base_llm,
            answer_max_words=answer_max_words,
            format_repair_max_words=format_repair_max_words,
        )
        self.peft_training_config = dict(peft_training_config)
        if "lora" not in self.peft_training_config or "training" not in self.peft_training_config:
            raise ValueError(
                "PEFTDebateAgent requires YAML-resolved lora and training configuration"
            )

    @staticmethod
    def _resolve_device_id(device_value: Any) -> int:
        """Resolve the CUDA device from the runtime configuration."""
        if isinstance(device_value, int):
            return int(device_value)

        text = str(device_value or "").strip()

        if text.startswith("cuda:"):
            return int(text.split(":", 1)[1])

        if text.isdigit():
            return int(text)

        raise ValueError(
            f"Unable to resolve CUDA device from runtime value: {device_value!r}"
        )

    def trainPEFT(self, feedback_dict: list[dict], save_directory: str) -> None:
        """Fine-tune on feedback rules that already passed the feedback gate.

        `feedback_dict` must contain non-empty ``{"text": ...}`` records.  The
        caller is responsible for ensuring these records passed the active feedback
        storage policy.  Quality/confidence may remain diagnostic references when
        YAML declares them not to be storage gates.
        """
        if not isinstance(self.base_llm, AnyHuggingFace):
            raise TypeError(
                "PEFTDebateAgent.trainPEFT requires an AnyHuggingFace base_llm"
            )

        clean_feedback = [
            {"text": str(item.get("text", "")).strip()}
            for item in feedback_dict
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        if not clean_feedback:
            raise ValueError("PEFT training received no accepted feedback text")

        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        class _NoParallelTrainer(Trainer):
            def _wrap_model(self, model, training=True, dataloader=None):
                return model

        llm = self.base_llm
        llm.logger.info(
            "[%s] Starting PEFT training for %s with %d accepted rules",
            self.name,
            llm.model_name,
            len(clean_feedback),
        )

        model_path = llm.model_path
        model_name = llm.model_name
        use_chat = llm._use_chat_str
        prompt_template = llm.prompt_template
        pipe_kwargs = dict(llm.kwargs)

        device_id = self._resolve_device_id(pipe_kwargs.get("device"))
        device_str = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)

        # Release inference weights before loading trainable weights.
        llm.model.cpu()
        for attr in ("model", "tokenizer", "llm", "pipe"):
            if hasattr(llm, attr):
                delattr(llm, attr)
        gc.collect()
        torch.cuda.empty_cache()

        hf_cache = os.environ.get("HF_HUB_CACHE")
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=hf_cache, token=hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        peft_cfg = self.peft_training_config
        lora_cfg = dict(peft_cfg["lora"])
        training_cfg = dict(peft_cfg["training"])

        use_bf16 = bool(torch.cuda.is_bf16_supported())
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        train_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=hf_cache,
            token=hf_token,
            torch_dtype=dtype,
            device_map={"": device_str},
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            target_modules=str(lora_cfg["target_modules"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=str(lora_cfg["bias"]),
        )
        train_model.config.use_cache = False
        train_model.enable_input_require_grads()
        train_model = get_peft_model(train_model, lora_config)
        train_model.gradient_checkpointing_enable()

        def _tokenize(examples):
            tokenized = tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=int(training_cfg["max_length"]),
            )
            labels = []
            for input_ids, attention_mask in zip(
                tokenized["input_ids"], tokenized["attention_mask"]
            ):
                labels.append(
                    [token if mask else -100 for token, mask in zip(input_ids, attention_mask)]
                )
            tokenized["labels"] = labels
            return tokenized

        dataset = Dataset.from_list(clean_feedback).map(_tokenize, batched=True)
        training_args = TrainingArguments(
            output_dir=save_directory,
            num_train_epochs=float(training_cfg["num_train_epochs"]),
            per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
            learning_rate=float(training_cfg["learning_rate"]),
            bf16=use_bf16,
            fp16=not use_bf16,
            logging_steps=int(training_cfg["logging_steps"]),
            save_strategy=str(training_cfg["save_strategy"]),
            report_to=str(training_cfg["report_to"]),
        )
        trainer = _NoParallelTrainer(
            model=train_model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        merged = train_model.merge_and_unload()
        os.makedirs(save_directory, exist_ok=True)
        merged.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        llm.logger.info(
            "[%s] PEFT training complete; merged model saved to %s",
            self.name,
            save_directory,
        )

        del train_model, merged, tokenizer, trainer, dataset
        gc.collect()
        torch.cuda.empty_cache()

        # Recreate the inference wrapper with the original generation settings.
        pipe_kwargs["device"] = device_str
        self.base_llm = AnyHuggingFace(
            model_name=save_directory,
            use_chat=use_chat,
            prompt_template=prompt_template,
            **pipe_kwargs,
        )


class EvalAgent:
    """Independent response-only safety evaluator.

    The evaluator judges only the current agent response text itself. The original
    question is intentionally ignored for safety classification. It also receives no
    short-term state, long-term memory, self-assessment, historical reliability,
    peer metadata, critique context, or previous evaluator result.

    ``question_text`` remains in ``run`` only for call-site compatibility with the
    existing debate orchestrator. It must never be rendered into evaluator input.
    """

    def __init__(
        self,
        name: str,
        base_llm: LLMType | LlamaGuard | None,
    ) -> None:
        self.name = str(name)
        self.base_llm = base_llm

    def run(self, question_text: str, agent_response: str) -> AgentOutput:
        # Protocol invariant: evaluator input is the current response text only.
        # Keep question_text in the signature to avoid changing the debate call chain,
        # but intentionally discard it here.
        del question_text
        response_text = str(agent_response)

        if isinstance(self.base_llm, LlamaGuard):
            return self.base_llm(response_text)

        if isinstance(
            self.base_llm,
            (AnyOpenAILLM, AnyHuggingFace, AnyVLLM, AnyGoogleGenerativeAI),
        ):
            return self.base_llm(
                {
                    "agent_response": response_text,
                }
            )

        if self.base_llm is None:
            try:
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                response = client.moderations.create(
                    model="omni-moderation-latest",
                    input=response_text,
                )
                # Preserve the existing textual convention consumed by the parser.
                return "unsafe" if response.results[0].flagged else "safe"
            except Exception as exc:
                # Fail closed. The parser will treat this as unsafe rather than
                # silently allowing content when moderation is unavailable.
                return f"unsafe\nMODERATION_ERROR:{type(exc).__name__}"

        raise TypeError(f"Unsupported evaluator base_llm type: {type(self.base_llm)!r}")


class FeedbackAgent:
    """Generate reusable correction rules from concrete observed failures."""

    def __init__(self, name: str, base_llm: LLMType) -> None:
        self.name = str(name)
        self.base_llm = base_llm

    def run(self, question_text: str, failure_cases: str) -> AgentOutput:
        return self.base_llm(
            {
                "question": str(question_text),
                "chat_history": str(failure_cases),
            }
        )
