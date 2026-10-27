# CaRDebate

CaRDebate: Decentralized Safety-Aware Critique and Revision in Multi-Agent Debate（Submitted to WWW 2027）

## Why debate?

Multi-agent debate (MAD) offers a novel perspective for LLM safety by preserving the independence of participating models and enabling agents with complementary capabilities to scrutinize and refine one another's responses, thereby identifying potential risks and correcting failures overlooked by individual models.

## datasets

```text
    HarmBench: "/datasets/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
    BeaverTails-Evaluation: "/datasets/PKU-Alignment/BeaverTails-Evaluation/v1/beavertails_evaluation.csv"
    CoSafe: "/datasets/CoSafe-Dataset/cosafe.csv"
    Aegis 2: "/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0/aegis2_core.csv"
    WildJailbreak: "/datasets/allenai/wildjailbreak/train/train.csv"
```

## models

```text
    Group 1
    Mistral-7B-Instruct-v0.2: "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2"
    Llama-3.2-3B-Instruct: "https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct"
    Phi-3.5-mini-instruct: "https://huggingface.co/microsoft/Phi-3.5-mini-instruct"


    Group 2
    Gemma-3-12B-IT: "https://huggingface.co/google/gemma-3-12b-it"
    Qwen3-8B: "https://huggingface.co/Qwen/Qwen3-8B"
    DeepSeek-R1-Distill-Llama-8B: "https://huggingface.co/DeepSeekAI/DeepSeek-R1-Distill-Llama-8B"


    Group 3
    C4AI Command R7B 12-2024: "https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024"
    GLM-4-9B-Chat-HF: "https://huggingface.co/zai-org/glm-4-9b-chat-hf"
    InternLM3-8B-Instruct: "https://huggingface.co/internlm/internlm3-8b-instruct"


    Group 4
    Granite-3.3-8B-Instruct: "https://huggingface.co/ibm-granite/granite-3.3-8b-instruct"
    Hunyuan-7B-Instruct: "https://huggingface.co/tencent/Hunyuan-7B-Instruct"
    Falcon-H1-7B-Instruct: "https://huggingface.co/tiiuae/Falcon-H1-7B-Instruct"


    safety evaluator
    Llama-Guard-3-8B: "https://huggingface.co/meta-llama/Llama-Guard-3-8B"


    feedback generator: GPT-4o-mini
```


## Framework of CaRDebate

![1793062569117](image/Readme/1793062569117.png)

## Main Implementation Files

- `CaRDebate/run.py`: `init_agents()`, `run_debate()`, remote evaluator adapter, and per-question exception handling.
- `CaRDebate/agents.py`: `DebateAgent.run()`, `EvalAgent.run()`; answer-stage execution and evaluator invocation adapter.
- `CaRDebate/debate.py`: `Debate.start()`, `_run_initial_round()`, `_run_revision_round()`, `_evaluate_round()`, `_run_current_evaluator()`; safe submission and failure handling.
- `CaRDebate/llm.py`: Model wrapper and OpenAI chat JSON schema selection.
- `CaRDebate/debate_prompts.py`: Critique and feedback prompts; initial-round and evaluator prompts are sourced from the shared protocol.
- `CaRDebate/confidence.py`: CaR adaptation for shared answers and evaluator parsing, plus label utility functions.
- `debate_answer_protocol.py`: Shared initial-round answer/repair prompts, JSON schemas, answer parsing, and evaluator prompts/parsers used by both sides.
- `CaRDebate/utils.py`: YAML loading, validation, and generation runtime parameter expansion.

## Create conda environment

```text
   conda env create -f environment.yml
```

## Model deployment

```text
bash ..\cardebate_assets\code\my_model_server_scripts\group1\run_model_server.sh
```

## run CaRDebate

```text
bash ..\cardebate_assets\code\my_debate_scripts\run_cardebate_all_datasets_group1.sh
```
