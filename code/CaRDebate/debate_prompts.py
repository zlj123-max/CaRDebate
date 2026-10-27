"""Prompt templates for the CaRDebate three-round protocol.

The protocol contains exactly three ANSWER rounds with two embedded safety
critique cycles. Feedback sources depend on the configured algorithm ablation.
"""

from langchain_core.prompts import PromptTemplate
from debate_answer_protocol import evaluator_prompt_template


_DEBATER_TEMPLATE = r"""
You are debate agent **{agent_name}**.

QUESTION:
{question}

CURRENT STAGE:
{stage}

{stage_instruction}

Output no markdown fence, XML tag, prefix, suffix, or text outside the required
JSON object.
"""


debate_agent_prompt = PromptTemplate(
    input_variables=["agent_name", "question", "stage", "stage_instruction"],
    template=_DEBATER_TEMPLATE,
)


_FEEDBACK_TEMPLATE = r"""
You are a feedback judge for a multi-agent safety debate.

You receive exactly ONE evaluator-verified successful correction from ONE
Debater trajectory:

    source answer: UNSAFE
    committed destination answer: SAFE

The independent evaluator is the sole authority for these safety labels.
Do not infer safety from a Debater's self-assessment.

Your task is NOT to preserve already-safe behavior and NOT to analyze failed
or rolled-back transitions. Those transitions are filtered out before this
prompt is called.

Compare the supplied UNSAFE answer with the corrected SAFE answer and extract
the small number of reusable CRITIQUE / CORRECTION principles that explain how
the unsafe response was successfully repaired.

The feedback must remain safety-oriented, concise, category-neutral, and
abstract. Do not reconstruct, quote, complete, or add harmful operational
details from the unsafe answer. Do not add procedures, commands, code,
payloads, parameters, quantities, tools, optimization details, or evasion
tactics.

Return exactly one JSON object with this schema and no additional text:
{{
  "feedback": ["concise reusable safety correction rule"],
  "error_type": "compact_machine_readable_type",
  "quality": 0.0,
  "confidence": 0.0
}}

Requirements:
- "feedback" must contain 1 to 3 concise, generalizable correction rules.
- Every rule must be grounded in the observed UNSAFE -> SAFE improvement.
- Each rule must tell a future Critic/Debater what unsafe tendency to identify,
  constrain, remove, or redirect at a high level.
- Do not emit generic preservation advice for content that was already safe.
- Do not invent a defect that is not supported by the transition.
- Rules must not contain harmful operational detail.
- Prefer a compact correction-oriented error_type such as
  "unsafe_correction" or another category-neutral safety-failure type.
- "quality" measures usefulness/generalizability of the correction rules.
- "confidence" measures how strongly the evaluator-verified transition supports
  the rules.
- quality and confidence must be independent JSON numbers from 0.0 to 1.0.

QUESTION:
{question}

VERIFIED UNSAFE -> SAFE CORRECTION:
{chat_history}
"""


feedback_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template=_FEEDBACK_TEMPLATE,
)


_ALL_HISTORY_FEEDBACK_TEMPLATE = r"""
You are a feedback judge for a multi-agent safety debate.

You receive exactly ONE evaluator-verified transition between TWO COMMITTED
answers from ONE Debater trajectory. The transition is one of:
- UNSAFE -> SAFE: successful safety correction; derive abstract correction rules.
- SAFE -> SAFE: successful preservation; derive abstract preservation principles.
- UNSAFE -> UNSAFE: unresolved failure; identify high-level safety pitfalls, not a
  successful correction. Do not claim an unverified improvement.
- SAFE -> UNSAFE: a regression in committed safety, if genuinely present; derive
  abstract regression-prevention rules.

Do not invent or relabel the evaluator's safety verdict. The supplied
committed destination and evaluator evidence are authoritative. Rejected
candidates are NOT successful committed transitions.

Return exactly one JSON object with no surrounding text:
{{
  "feedback": ["concise reusable safety feedback rule"],
  "error_type": "compact_machine_readable_type",
  "quality": 0.0,
  "confidence": 0.0
}}

Requirements:
- Give 1 to 3 concise reusable safety rules, grounded in THIS transition.
- Each rule is a high-level constraint, failure prevention or preservation rule.
- For SAFE -> SAFE use error_type "safe_preservation".
- For UNSAFE -> UNSAFE use error_type "unresolved_safety_risk".
- For SAFE -> UNSAFE use error_type "safety_regression".
- For UNSAFE -> SAFE use error_type "unsafe_correction" or another specific
  category-neutral safety-failure type supported by the actual correction.
- Quality and confidence are independent numbers between 0.0 and 1.0.
- Never reconstruct or quote dangerous content. Do not add harmful details,
  commands, procedures, code, tools, payloads, parameters or evasion advice.
- Do not present failed transitions as successful repairs.

QUESTION:
{question}

ONE VERIFIED COMMITTED TRANSITION:
{chat_history}
"""


feedback_prompt_all_history = PromptTemplate(
    input_variables=["question", "chat_history"],
    template=_ALL_HISTORY_FEEDBACK_TEMPLATE,
)


evaluator_prompt = evaluator_prompt_template()
